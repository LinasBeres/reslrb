from dataclasses import dataclass

import drjit as dr
import mitsuba as mi


@dataclass
class Reservoir:
    """The one sensor connection kept for each light path."""
    ds: mi.DirectionSample3f
    w_raw: mi.Spectrum
    pi: mi.PreliminaryIntersection3f
    pi_prev: mi.PreliminaryIntersection3f
    ray: mi.Ray3f
    ray_prev: mi.Ray3f
    si: mi.SurfaceInteraction3f
    depth: mi.UInt32
    w_sel: mi.Float
    throughput: mi.Spectrum
    sensor_w: mi.Spectrum
    ray_state_jacobian: mi.Matrix4f

    @staticmethod
    def empty():
        return Reservoir(
            ds=dr.zeros(mi.DirectionSample3f),
            w_raw=mi.Spectrum(0),
            pi=dr.zeros(mi.PreliminaryIntersection3f),
            pi_prev=dr.zeros(mi.PreliminaryIntersection3f),
            ray=dr.zeros(mi.Ray3f),
            ray_prev=dr.zeros(mi.Ray3f),
            si=dr.zeros(mi.SurfaceInteraction3f),
            depth=mi.UInt32(0),
            w_sel=mi.Float(0),
            throughput=mi.Spectrum(0),
            sensor_w=mi.Spectrum(0),
            ray_state_jacobian=mi.Matrix4f(0.0),
        )

    def stream_update(self, replace, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, dr.select(replace, value, getattr(self, name)))
