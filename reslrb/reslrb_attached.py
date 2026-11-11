import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import solid_angle_to_area_jacobian

from .base import LightTracerBase
from .jacobian import TwoPointParametrization, barycentric, jacobian, safe_inverse
from .reservoir import Reservoir


class ResLRBAttachedIntegrator(LightTracerBase):
    """Reservoir Light Replay Backpropagation, attached version.

    Same reservoir and replay as `reslrb`, but the sensor position of the kept
    connection also depends on the scene. Each segment is parametrised by the
    barycentric coordinates of its two endpoints. The primal pass chains the
    per-vertex Jacobians of these parameters and caches the chain at the kept
    vertex. The adjoint pass maps the gradient of the sensor position back to
    each vertex through the inverse of the chain.
    """

    def __init__(self, props):
        super().__init__(props)
        self.noise_sigma = dr.opaque(mi.Float, props.get('noise_sigma', 1e-2))

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()

        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())
            sample_scale = dr.prod(film.crop_size()) / (spp * dr.prod(film.size()))

            L, pos, depth, valid, path_jacobian = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sensor=sensor,
                sampler=sampler.clone(), active=mi.Bool(True), compute_jacobian=True)

            dL, dpos = self.splat_backward(film, grad_in, pos, L, valid, sample_scale)

            self.sample(
                mode=dr.ADMode.Backward, scene=scene, sensor=sensor, sampler=sampler,
                active=valid, depth_reached=depth, dL=dL, dpos=dpos, state_in=L,
                path_jacobian=path_jacobian, compute_jacobian=True)
            dr.eval()

    def regularise(self, J, sampler):
        """Small random shear, so that the Jacobian at diffuse vertices stays invertible."""
        val = self.noise_sigma * (sampler.next_1d() - 0.5)
        noise = mi.Matrix4f(0.0)
        noise[2, 0] = val
        noise[3, 1] = val
        return J + noise

    @dr.syntax
    def sample(self, mode, scene, sensor, sampler, active,
               depth_reached=None, dL=None, dpos=None, state_in=None, path_jacobian=None,
               compute_jacobian=False):
        primal = mode == dr.ADMode.Primal
        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)

        active = mi.Bool(active)
        L = mi.Spectrum(0 if primal else state_in)
        dL = mi.Spectrum(dL if dL is not None else 0)
        dpos = mi.Point2f(dpos if dpos is not None else 0)
        path_jacobian = mi.Matrix4f(path_jacobian if path_jacobian is not None else 0)
        # Adjoint of the sensor position, written in the parameters of the kept segment
        pos_adjoint = mi.Vector4f(0.0, 0.0, dpos.x, dpos.y) @ path_jacobian

        depth = mi.UInt32(1)
        throughput = mi.Spectrum(1)
        η = mi.Spectrum(1)
        res = Reservoir.empty()
        w_sum = mi.Float(0)
        ray_state_jacobian = mi.Matrix4f(1.0)

        ray_weight, ray = self.sample_emitter(scene, sensor, sampler)

        ray_prev = mi.Ray3f(ray)
        ray_prev.d = -ray.d
        pi_prev = scene.ray_intersect_preliminary(ray_prev, active)
        pi = scene.ray_intersect_preliminary(ray, coherent=True, reorder=False, active=active)
        si_prev = pi_prev.compute_surface_interaction(
            ray_prev, ray_flags=mi.RayFlags.Minimal, active=active)

        active &= pi.is_valid() & (depth < self.max_depth)
        active &= dr.max(mi.unpolarized_spectrum(ray_weight)) != 0

        while dr.hint(active, max_iterations=self.max_depth, mode='symbolic',
                      label=f'ResLRB attached {mode.name}'):
            active_next = mi.Bool(active)

            ray_params = TwoPointParametrization(
                pi_prev.prim_uv, pi_prev.is_valid(), pi.prim_uv, pi.is_valid(), ray.d)
            ray_params_list = ray_params.params_list()

            with dr.resume_grad(*ray_params_list, when=compute_jacobian):
                if dr.hint(compute_jacobian, mode='scalar'):
                    dr.enable_grad(*ray_params_list)
                    ray_attached = ray_params.spawn_ray(si_prev, pi)
                else:
                    ray_attached = ray
                si = pi.compute_surface_interaction(
                    ray_attached, ray_flags=mi.RayFlags.All, active=active_next)
                bsdf = si.bsdf(ray_attached)

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
                    replace, ds=sensor_ds, w_raw=W_candidate, pi=pi, pi_prev=pi_prev,
                    ray=ray, ray_prev=ray_prev, depth=depth, w_sel=w,
                    throughput=throughput, sensor_w=sensor_w,
                    ray_state_jacobian=ray_state_jacobian)

            # BSDF sampling. The samples are reused by the adjoint below.
            bsdf_d_sample = sampler.next_1d()
            bsdf_p_sample = sampler.next_2d()

            with dr.resume_grad(*ray_params_list, ray_attached, si, bsdf, when=compute_jacobian):
                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, bsdf_d_sample, bsdf_p_sample, active_next)
                wi_dot_n = dr.dot(si.n, -ray_attached.d)
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

            with dr.resume_grad(*ray_params_list, ray_attached, si, bsdf_sample, when=compute_jacobian):
                ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(
                ray_next, coherent=False, reorder=False, active=active_next)
            active_next &= pi.is_valid()

            # Jacobian of this segment's parameters to the next one's
            if dr.hint(compute_jacobian, mode='scalar'):
                with dr.resume_grad(*ray_params_list, ray_attached, ray_next):
                    uv_current = barycentric(pi, ray_attached)
                    uv_next = barycentric(pi_next, ray_next)
                    next_params = mi.Vector4f(uv_current.x, uv_current.y, uv_next.x, uv_next.y)
                    J_segment = jacobian(ray_params_list, next_params)
                ray_state_jacobian = self.regularise(J_segment, sampler) @ ray_state_jacobian

            # Adjoint, up to the kept vertex
            if dr.hint(not primal, mode='scalar'):
                diff_active = depth <= depth_reached
                si_next = pi_next.compute_surface_interaction(
                    ray_next, ray_flags=mi.RayFlags.Minimal, active=diff_active)

                with dr.resume_grad():
                    si_d = pi.compute_surface_interaction(
                        ray, ray_flags=mi.RayFlags.All, active=diff_active)
                    bs_d, bs_weight_d = si_d.bsdf(ray).sample(
                        bsdf_ctx, si_d, bsdf_d_sample, bsdf_p_sample, diff_active)

                    # Radiometric part, as in the detached version
                    J = solid_angle_to_area_jacobian(si.p, si_next.p, si_next.n, diff_active)
                    wo_world = dr.replace_grad(
                        ray_next.d,
                        dr.select(diff_active, dr.normalize(si_next.p - si.p), ray_next.d))
                    wo = si.to_local(wo_world)
                    wi_dot_n = dr.dot(si.n, si.to_world(si.wi))
                    wo_dot_n = dr.dot(si.n, wo_world)
                    bs_weight_d *= dr.abs((mi.Frame3f.cos_theta(si.wi) * wo_dot_n) /
                                          (mi.Frame3f.cos_theta(wo) * wi_dot_n))

                    Lr = L * dr.relative_grad(bs_weight_d) * dr.relative_grad(J)
                    if dr.hint(dr.grad_enabled(Lr), mode='scalar'):
                        dr.backward_from(dL * Lr, flags=dr.ADFlag.ClearVertices)

                    # Positional part: the sensor position adjoint, moved into this
                    # vertex's frame through the inverse of the accumulated Jacobian
                    dr.disable_grad(bs_weight_d)
                    ray_next_d = si_d.spawn_ray(si_d.to_world(bs_d.wo))
                    end_uv = barycentric(pi_next, ray_next_d)
                    local_adjoint = pos_adjoint @ safe_inverse(ray_state_jacobian)

                    if dr.hint(dr.grad_enabled(end_uv), mode='scalar'):
                        grad = mi.Vector2f(local_adjoint.z, local_adjoint.w)
                        dr.set_grad(end_uv, dr.select(diff_active, grad, mi.Point2f(0.0, 0.0)))
                        dr.enqueue(dr.ADMode.Backward, end_uv)
                        dr.traverse(dr.ADMode.Backward)

            reorder_hint = dr.reinterpret_array(mi.UInt32, pi.shape)
            reorder_hint = (reorder_hint << 1) | dr.select(active_next, 1, 0)
            depth = dr.reorder_threads(reorder_hint, 16, depth)

            active = active_next
            pi_prev = pi
            pi = pi_next
            si_prev = dr.detach(si)
            ray_prev = dr.detach(ray)
            ray = dr.detach(ray_next)

        valid = res.depth != 0
        pdf_ratio = dr.select(res.w_sel > 0, w_sum / res.w_sel, mi.Float(0))

        # Close the chain with the Jacobian of the sensor connection
        if dr.hint(compute_jacobian and primal, mode='scalar'):
            si_prev = res.pi_prev.compute_surface_interaction(
                res.ray_prev, ray_flags=mi.RayFlags.Minimal, active=valid)
            ray_params = TwoPointParametrization(
                res.pi_prev.prim_uv, res.pi_prev.is_valid(), res.pi.prim_uv,
                res.pi.is_valid(), res.ray.d)
            ray_params_list = ray_params.params_list()
            with dr.resume_grad(*ray_params_list):
                dr.enable_grad(*ray_params_list)
                ray_attached = ray_params.spawn_ray(si_prev, res.pi)
                si_attached = res.pi.compute_surface_interaction(
                    ray_attached, ray_flags=mi.RayFlags.All, active=valid)
                sensor_ds_d, _ = sensor.sample_direction(
                    it=si_attached, sample=mi.Point2f(0.5, 0.5), active=valid)
                uv_current = barycentric(res.pi, ray_attached)
                next_params = mi.Vector4f(uv_current.x, uv_current.y,
                                          sensor_ds_d.uv.x, sensor_ds_d.uv.y)
                J_sensor = jacobian(ray_params_list, next_params)
            ray_state_jacobian = J_sensor @ res.ray_state_jacobian

        # The kept connection, reweighted by W / w
        with dr.resume_grad(when=not primal):
            si = res.pi.compute_surface_interaction(res.ray, ray_flags=mi.RayFlags.All, active=valid)
            if dr.hint(not primal, mode='scalar'):
                # Only sensors that do not sample a lens are supported.
                sensor_ds, sensor_w = sensor.sample_direction(
                    it=si, sample=mi.Point2f(0.5, 0.5), active=valid)
                res.sensor_w = dr.replace_grad(res.sensor_w, sensor_w / dr.detach(sensor_w))
                res.ds = dr.replace_grad(res.ds, sensor_ds)

            d = dr.normalize(res.ds.p - si.p)
            local_d = si.to_local(d)
            wi_dot_n = dr.dot(si.n, si.to_world(si.wi))
            wo_dot_n = dr.dot(si.n, d)
            no_leak = (wi_dot_n * mi.Frame3f.cos_theta(si.wi) > 0.0) & \
                      (wo_dot_n * mi.Frame3f.cos_theta(local_d) > 0.0)
            correction = dr.select(no_leak,
                                   dr.abs((mi.Frame3f.cos_theta(si.wi) * wo_dot_n) /
                                          (mi.Frame3f.cos_theta(local_d) * wi_dot_n)),
                                   0.0)
            bsdf_val = si.bsdf(res.ray).eval(bsdf_ctx, si, local_d, valid)

            L = pdf_ratio * ray_weight * res.throughput * res.sensor_w * bsdf_val * correction

            if dr.hint(not primal and dr.grad_enabled(L), mode='scalar'):
                dr.backward_from(dL * L, flags=dr.ADFlag.ClearVertices)

        return L if primal else dL, res.ds.uv, res.depth, valid, ray_state_jacobian


mi.register_integrator("reslrb_attached", lambda props: ResLRBAttachedIntegrator(props))
