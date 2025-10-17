from vsdenoise import bm3d, mc_degrain, nl_means
from pydantic import BaseModel, ConfigDict
from vsdeband import placebo_deband
from vskernels import Lanczos
from vsmasktools import Morpho
import vsmlrt
from muxtools import Setup
from muxtools import mux as vsmux
from vsmuxtools import do_audio, settings_builder_x265, SourceFilter, src_file, x265
from vspreview import is_preview
from vsscale import Rescale, descale_error_mask
from vstools import core, depth, DitherType, finalize_clip, get_y, join, set_output, SPath, vs

from .insaneAAMod import insaneAA
from .vodesfuncNoiseMod import adaptive_grain, ntype4


class FilterchainResults(BaseModel):
    src: vs.VideoNode
    final: vs.VideoNode
    audio: src_file

    model_config = ConfigDict(arbitrary_types_allowed=True)


def filterchain(source):
    amzn_file = src_file(source, preview_sourcefilter=SourceFilter.BESTSOURCE)
    src = amzn_file.init_cut()
    

    cclip = src.resize.Bicubic(filter_param_a=0, filter_param_b=0.5, \
                               width=1920, height=1088, src_left=0, src_top=-4, src_width=1920, src_height=1088, \
                               format=vs.RGBS, range=1) # 1886 transfer works the best
    cclip = vsmlrt.inference(cclip, SPath(vsmlrt.models_path) / "anime-segmentation" / "isnet_is.onnx", backend=vsmlrt.Backend.TRT(fp16=True))
    cclip = Morpho.maximum(cclip)
    cclip = cclip.akarin.Expr("""
                               x[-2,-5] x[-1,-5] x[0,-5] x[1,-5] x[2,-5]
                      x[-3,-4] x[-2,-4] x[-1,-4] x[0,-4] x[1,-4] x[2,-4] x[3,-4]
             x[-4,-3] x[-3,-3] x[-2,-3] x[-1,-3] x[0,-3] x[1,-3] x[2,-3] x[3,-3] x[4,-3]
    x[-5,-2] x[-4,-2] x[-3,-2] x[-2,-2] x[-1,-2] x[0,-2] x[1,-2] x[2,-2] x[3,-2] x[4,-2] x[5,-2]
    x[-5,-1] x[-4,-1] x[-3,-1] x[-2,-1] x[-1,-1] x[0,-1] x[1,-1] x[2,-1] x[3,-1] x[4,-1] x[5,-1]
    x[-5,0]  x[-4,0]  x[-3,0]  x[-2,0]  x[-1,0]  x[0,0]  x[1,0]  x[2,0]  x[3,0]  x[4,0]  x[5,0]
    x[-5,1]  x[-4,1]  x[-3,1]  x[-2,1]  x[-1,1]  x[0,1]  x[1,1]  x[2,1]  x[3,1]  x[4,1]  x[5,1]
    x[-5,2]  x[-4,2]  x[-3,2]  x[-2,2]  x[-1,2]  x[0,2]  x[1,2]  x[2,2]  x[3,2]  x[4,2]  x[5,2]
             x[-4,3]  x[-3,3]  x[-2,3]  x[-1,3]  x[0,3]  x[1,3]  x[2,3]  x[3,3]  x[4,3]
                      x[-3,4]  x[-2,4]  x[-1,4]  x[0,4]  x[1,4]  x[2,4]  x[3,4]
                               x[-2,5]  x[-1,5]  x[0,5]  x[1,5]  x[2,5]
    sort97 drop8 cluster! drop85 high! drop2
    high@ 0.95 > x 0.85 > and 1 x ? continue!
    cluster@ 0.10 > high@ 0.95 < and continue@ 0.75 < and continue@ 1.333333333333 * 3 pow 3 pow 0.75 * continue@ ? continue!
    continue@ 0.10 > continue@ 0 ?
    """)
    cclip = Morpho.dilation(cclip, radius=2)
    cclip = Morpho.inflate(cclip, radius=2)
    cclip = cclip.std.Crop(top=4, bottom=4)
    cclip = depth(cclip, 16, dither_type=DitherType.NONE)


    aa = insaneAA(src, descale_height=864, dehalo=True, alpha=0.75, beta=0.15, nrad=3, mdis=30)

    rescale = Rescale(src, height=864, width=1536, kernel=Lanczos(2)).rescale
    aa_mask = descale_error_mask(src, rescale, thr=0.01, expands=(5, 6, 1), blur=2, tr=2)
    aa_mask = core.akarin.Expr([cclip, aa_mask], "x y -")

    aa = core.std.MaskedMerge(src, aa, aa_mask)


    ref = mc_degrain(aa, tr=2, thsad=140)
    ref_y = get_y(ref)
    
    aa_y = get_y(aa)
    b_dn_y = bm3d(aa_y, ref=ref_y, sigma=0.7, tr=2, profile=bm3d.Profile.NORMAL)
    
    c_dn_y = bm3d(aa_y, ref=ref_y, sigma=2.2, tr=1, profile=bm3d.Profile.NORMAL)
    c_db_y = placebo_deband(c_dn_y, radius=24.0)

    dn_db_y = core.std.MaskedMerge(b_dn_y, c_db_y, cclip)

    dn_uv = nl_means(aa, ref=ref, h=0.3, tr=2, planes=[1, 2])

    dn_db = join(dn_db_y, dn_uv)


    final = adaptive_grain(dn_db, strength=[2.1, 0.42], size=3.26, temporal_average=50, seed=274810, **ntype4)

    final = finalize_clip(final)


    if is_preview():
        set_output(src, "src")
        set_output(aa, "aa")
        set_output(core.akarin.Expr([aa, src], ["x y - 10 * 32768 +"]), "aa")
        set_output(dn_db, "dn_db")
        set_output(core.akarin.Expr([dn_db, aa], ["x y - 10 * 32768 +"]), "dn_db")
        set_output(final, "final")
        set_output(core.akarin.Expr([depth(final, 16), dn_db], ["x y - 10 * 32768 +"]), "final")


    return FilterchainResults(src=src, final=final, audio=amzn_file)
    

def mux(episode, filterchain_results):
    setup = Setup(episode)

    settings = settings_builder_x265(hist_scenecut="", frames=filterchain_results.final.num_frames,
                                     crf=14.00, qcomp=0.71, rect=False)
    video = x265(settings, resumable=False, csv=SPath(setup.work_dir) / "x265_log.csv").encode(filterchain_results.final)

    audio = do_audio(filterchain_results.audio)

    return vsmux(video.to_track(lang="ja", args=["--deterministic", "274810"]),
               audio.to_track(lang="ja"))
