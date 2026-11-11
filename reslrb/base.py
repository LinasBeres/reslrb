import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import RBIntegrator


class LightTracerBase(RBIntegrator):
    """Shared parts of the ResLRB light tracers.

    Subclasses implement `sample`, which traces one light path per sample and
    returns its splatted value, its sensor position, the depth of the kept
    connection and whether a connection was kept.
    """

    def sample_emitter(self, scene, sensor, sampler):
        wavelength_sample = sampler.next_1d()
        position_sample = sampler.next_2d()
        direction_sample = sampler.next_2d()

        emitter_idx, emitter_weight, wavelength_sample = scene.sample_emitter(wavelength_sample)
        emitter = scene.emitters_dr()[emitter_idx]
        ray, ray_weight = emitter.sample_ray(
            sensor.shutter_open(), wavelength_sample, position_sample, direction_sample)
        return ray_weight * emitter_weight, dr.detach(ray)

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()

        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())
            sample_scale = dr.prod(film.crop_size()) / (spp * dr.prod(film.size()))

            out = self.sample(mode=dr.ADMode.Primal, scene=scene, sensor=sensor,
                              sampler=sampler, active=mi.Bool(True))
            L, pos, valid = out[0], out[1], out[3]

            block = film.create_block(size=mi.ScalarVector2u(0), normalize=True)
            block.set_coalesce(False)
            self.splat(block, film, pos, L * sample_scale, valid)
            film.put_block(block)
            return film.develop()

    def render_forward(self, scene, params, sensor=0, seed=0, spp=0):
        raise NotImplementedError("ResLRB only supports reverse mode")

    def splat(self, block, film, pos, value, valid):
        RBIntegrator._splat_to_block(block, film, pos + block.offset(), value=value, weight=0.0,
                                     alpha=dr.select(valid, 1.0, 0.0), aovs=[],
                                     wavelengths=mi.Spectrum(0))

    def splat_backward(self, film, grad_in, pos, L, valid, sample_scale):
        """Gradient of the loss with respect to each sample's value and sensor position."""
        block = film.create_block(size=mi.ScalarVector2u(0), normalize=True)
        with dr.resume_grad():
            L = mi.Spectrum(dr.detach(L))
            pos = mi.Point2f(dr.detach(pos))
            dr.enable_grad(L, pos)
            self.splat(block, film, pos, L * sample_scale, valid)
            film.put_block(block)
            image = film.develop()
            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(dr.ADMode.Backward)
            dL, dpos = dr.grad(L), dr.grad(pos)
            film.clear()
        return dL, dpos
