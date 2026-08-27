import subprocess
import argparse
import time 
import sys


parser = argparse.ArgumentParser(
    description="APEX Kirchhoff-Fresnel optimization")

parser.add_argument('-f', '--filename', dest='filename' , type=str,
                    help="Input filename should be a .reg or .hdf5 output of the fourier pipeline")

parser.add_argument("-geo_file", "--geo_file", dest='geo_file', type=str, default=None,
                    help="Geometry file (.npz). If None just use the default hyperparameters.")

parser.add_argument("-panel_file", "--panel_file", dest='panel_file', type=str, default=None,
                    help="Panel deformation file (.npz). If None just start the coefficients at zero")

parser.add_argument('-mi', '--max_iters', dest='max_iters', type=int, default=70,
                     help="maximum iterations of the optimization")

parser.add_argument("-no_stop", "--no_stop", dest="no_stop", action='store_true',
                    help="Avoids all the stop mechanism, the optimization runs up to the max iteration")
parser.add_argument("-plot_path", dest='plot_path', type=str, default="~/MODULES/physical_optics/")


geo_script = "python geometrical_fit.py --filename %s --max_iters %i -plot_path %s"(%args.filename, args.max_iters, args.plot_path)
panel_scritp = "python panel_fit.py --filename %s --max_iters %i -plot_path %s"%args.filename, args.max_ters, args.plot_path)


filename = os.path.basename(args.filename))
if(filename.endswith(".reg")):
    filename = filename.split(".reg")[0]
elif(filename.endswith(".hdf5")):
    filename = filename.split(".hdf5")[0]

opt_dir = os.path.join(args.plot_path, filename)

geo_param_head = os.path.join(opt_dir, 'geometry_fit')
geo_param_tail = "geo_params.npz"

panel_param_head = os.path.join(opt_dir, 'panel_fit')
panel_param_tail = "panels_params_flipped.npz"


start = time.time()

print("Running first round of optimizations")
rc = subprocess.run(geo_script.split(" "))
if(rc.returncode):
    print("return code!=0")
    sys.exit()
geo_param_iter = os.path.join(geo_param_head, "000", geo_param_tail)

panel_cmd = panel_script+" -geo_file "+str(geo_param_iter)
rc = subprocess.run(panel_cmd.split(" "))
if(rc.returncode):
    print("return code!=0")
    sys.exit()
panel_param_iter = os.path.join(panel_param_head, "000", panel_param_tail)

iter_time = (start-time.time())/60/60
print("First opt steps took %.4f hr"%(iter_time))

for i in range(1,2):
    iter_start = time.time()
    geo_cmd = geo_script+" -geo_file %s -panel_file %s"%(geo_param_iter, panel_param_iter)
    rc = subprocess.run(geo_cmd)
    if(rc.returncode):
        print("return code!=0")
        sys.exit()
    geo_param_iter = os.path.join(geo_param_head, "%03d"%i, geo_param_tail)

    panel_cmd = panel_script+" -geo_file %s -panel_file %s"%(geo_param_iter, panel_param_iter)
    rc = subprocess.run(panel_cmd.split(" "))
    if(rc.returncode):
        print("return code!=0")
        sys.exit()
    
    iter_time = (time.time()-iter_start)/60/60
    print("Iter took %.4f hrs"%iter_time)

total_time = (time.time()-start)/60/60
print("Total time spend: %.4f hrs"%total_time)























