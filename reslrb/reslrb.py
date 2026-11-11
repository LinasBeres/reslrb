import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import solid_angle_to_area_jacobian

from .base import LightTracerBase
from .reservoir import Reservoir


class ResLRBIntegrator(LightTracerBase):
    """Reservoir Light Replay Backpropagation, detached version.

    The primal pass streams every sensor connection of a light path through a
    single-item weighted reservoir and splats only the kept one, scaled by
    W / w. The adjoint pass replays the path from the same seed up to the kept
    vertex and backpropagates one scattering event at a time.
    """

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()

        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())
            sample_scale = dr.prod(film.crop_size()) / (spp * dr.prod(film.size()))

            L, pos, depth, valid = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sensor=sensor,
                sampler=sampler.clone(), active=mi.Bool(True))

            δL, _ = self.splat_backward(film, grad_in, pos, L, valid, sample_scale)

            self.sample(
                mode=dr.ADMode.Backward, scene=scene, sensor=sensor, sampler=sampler,
                active=valid, depth_reached=depth, δL=δL, state_in=L)
            dr.eval()

    @dr.syntax
    def sample(self, mode, scene, sensor, sampler, active,
               depth_reached=None, δL=None, state_in=None):
        primal = mode == dr.ADMode.Primal
        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)

        L = mi.Spectrum(0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        active = mi.Bool(active)

        res = Reservoir.empty()
        depth = mi.UInt32(1)
        throughput = mi.Spectrum(1)
        η = mi.Spectrum(1)
        w_sum = mi.Float(0)

        ray_weight, ray = self.sample_emitter(scene, sensor, sampler)

        # The emitter vertex, so that the first segment can follow its motion.
        ray_prev = mi.Ray3f(ray)
        ray_prev.d = -ray.d
        pi_prev = scene.ray_intersect_preliminary(ray_prev, active)
        pi = scene.ray_intersect_preliminary(ray, coherent=True, reorder=False, active=active)

        active &= pi.is_valid() & (depth < self.max_depth)
        active &= dr.max(mi.unpolarized_spectrum(ray_weight)) != 0

        while dr.hint(active, max_iterations=self.max_depth, label=f'ResLRB {mode.name}'):
            active_next = mi.Bool(active)

            with dr.resume_grad(when=not primal):
                si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)
                if dr.hint(not primal, mode='scalar'):
                    si_prev = pi_prev.compute_surface_interaction(
                        ray_prev, ray_flags=mi.RayFlags.Minimal, active=active_next)
                    si_detached = dr.detach(si)
                    wi = dr.normalize(si_prev.p - si_detached.p)
                    si.wi = dr.replace_grad(si.wi, si_detached.to_local(wi))

            bsdf = si.bsdf(ray)

            # Sensor connection, streamed through the reservoir
            active_sensor = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & active_next
            sensor_ds, sensor_w = sensor.sample_direction(
                it=si, sample=sampler.next_2d(), active=active_sensor)
            W_candidate = sensor_w * throughput

            active_sensor &= (sensor_ds.pdf > 0.0) & dr.any(mi.unpolarized_spectrum(W_candidate) != 0.0)
            active_sensor &= ~scene.ray_test(si.spawn_ray_to(sensor_ds.p), active_sensor)

            w = dr.mean(mi.unpolarized_spectrum(W_candidate))
            w_sum = dr.select(active_sensor, w_sum + w, w_sum)
            replace = active_sensor & (sampler.next_1d() < w / w_sum)

            if replace:
                res.stream_update(
                    replace, ds=sensor_ds, w_raw=W_candidate, si=si, pi=pi,
                    pi_prev=pi_prev, ray=ray, ray_prev=ray_prev, depth=depth, w_sel=w)

            # BSDF sampling
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            wi_dot_n = dr.dot(si.n, -ray.d)
            wo_dot_n = dr.dot(si.n, si.to_world(bsdf_sample.wo))
            active_next &= (wi_dot_n * mi.Frame3f.cos_theta(si.wi) > 0.0) & \
                           (wo_dot_n * mi.Frame3f.cos_theta(bsdf_sample.wo) > 0.0)
            correction = dr.abs((mi.Frame3f.cos_theta(si.wi) * wo_dot_n) /
                                (mi.Frame3f.cos_theta(bsdf_sample.wo) * wi_dot_n))
            throughput *= bsdf_weight * correction
            η *= bsdf_sample.eta

            # Termination
            depth += 1
            active_next &= depth < self.max_depth
            β_max = dr.max(mi.unpolarized_spectrum(throughput))
            active_next &= β_max != 0

            rr_prob = dr.minimum(β_max * η**2, .95)
            rr_active = depth > self.rr_depth
            throughput[rr_active] *= dr.rcp(rr_prob)
            active_next &= ~rr_active | dr.any(sampler.next_1d() < rr_prob)

            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(
                ray_next, coherent=False, reorder=False, active=active_next)
            active_next &= pi.is_valid()

            # Adjoint: one scattering event at a time, up to the kept vertex
            if dr.hint(not primal, mode='scalar'):
                active_next = depth <= depth_reached
                si_next = pi_next.compute_surface_interaction(
                    ray_next, ray_flags=mi.RayFlags.Minimal, active=active_next)

                with dr.resume_grad():
                    J = solid_angle_to_area_jacobian(si.p, si_next.p, si_next.n, active_next)

                    wo_world = dr.replace_grad(
                        ray_next.d,
                        dr.select(active_next, dr.normalize(si_next.p - si.p), ray_next.d))
                    wo = si.to_local(wo_world)
                    wi_dot_n = dr.dot(si.n, si.to_world(si.wi))
                    wo_dot_n = dr.dot(si.n, wo_world)
                    correction = dr.abs((mi.Frame3f.cos_theta(si.wi) * wo_dot_n) /
                                        (mi.Frame3f.cos_theta(wo) * wi_dot_n))
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next) * correction

                    # L / f_s * d f_s, through the relative gradient
                    Lr = L * dr.relative_grad(bsdf_val) * dr.relative_grad(J)
                    if dr.hint(dr.grad_enabled(Lr), mode='scalar'):
                        dr.backward_from(δL * Lr)

            reorder_hint = dr.reinterpret_array(mi.UInt32, pi.shape)
            reorder_hint = (reorder_hint << 1) | dr.select(active_next, 1, 0)
            depth = dr.reorder_threads(reorder_hint, 16, depth)

            active = active_next
            pi_prev = pi
            pi = pi_next
            ray_prev = ray
            ray = ray_next

        # The kept connection, reweighted by W / w
        pdf_ratio = dr.select(res.w_sel > 0, w_sum / res.w_sel, mi.Float(0))
        valid = res.depth != 0

        with dr.resume_grad(when=not primal):
            if dr.hint(not primal, mode='scalar'):
                si_prev = res.pi_prev.compute_surface_interaction(
                    res.ray_prev, ray_flags=mi.RayFlags.Minimal, active=valid)
                si = res.pi.compute_surface_interaction(
                    res.ray, ray_flags=mi.RayFlags.All, active=valid)
                res.si = dr.replace_grad(res.si, si)

                si_detached = dr.detach(res.si)
                wi = dr.normalize(si_prev.p - si_detached.p)
                res.si.wi = dr.replace_grad(res.si.wi, si_detached.to_local(wi))

                # Only sensors that do not sample a lens are supported.
                _, sensor_w = sensor.sample_direction(
                    it=res.si, sample=mi.Point2f(0.5, 0.5), active=valid)
                res.w_raw = dr.replace_grad(res.w_raw, sensor_w / dr.detach(sensor_w))

            d = dr.normalize(res.ds.p - res.si.p)
            local_d = res.si.to_local(d)
            wi_dot_n = dr.dot(res.si.n, res.si.to_world(res.si.wi))
            wo_dot_n = dr.dot(res.si.n, d)
            no_leak = (wi_dot_n * mi.Frame3f.cos_theta(res.si.wi) > 0.0) & \
                      (wo_dot_n * mi.Frame3f.cos_theta(local_d) > 0.0)
            correction = dr.select(no_leak,
                                   dr.abs((mi.Frame3f.cos_theta(res.si.wi) * wo_dot_n) /
                                          (mi.Frame3f.cos_theta(local_d) * wi_dot_n)),
                                   0.0)
            bsdf_val = res.si.bsdf(res.ray).eval(bsdf_ctx, res.si, local_d, valid)

            L = res.w_raw * pdf_ratio * ray_weight * bsdf_val * correction

            if dr.hint(not primal and dr.grad_enabled(L), mode='scalar'):
                dr.backward_from(δL * L)

        return L if primal else δL, res.ds.uv, res.depth, valid


mi.register_integrator("reslrb", lambda props: ResLRBIntegrator(props))
