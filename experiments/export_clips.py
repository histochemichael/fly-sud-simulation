"""Export full-height condition clips and the additional DAN control example."""
import subprocess
import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg
from build_film import OUT,Clip,comparison_panel
for phase,start,duration in [('training',28,30),('test',58,20)]:
    for col,c in enumerate(['untrained','paired','unpaired']):
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-ss',str(start),'-i',str(OUT/'fly_ethanol_reward_silent.mp4'),'-t',str(duration),'-vf',f'crop=608:828:{32+624*col}:200','-c:v','libx264','-crf','19','-preset','fast','-an',str(OUT/f'{c}_{phase}.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
clips={'dan_silenced':Clip('dan_silenced')}
for phase,duration in [('training',30),('test',20)]:
    writer=iio.get_writer(OUT/f'dan_silenced_{phase}.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
    for frame in range(duration*25):
        im=comparison_panel(clips,phase,frame/25*.1,['dan_silenced']).crop((32,200,640,1028))
        writer.append_data(np.asarray(im))
    writer.close()
    comparison_panel(clips,phase,2.6 if phase=='training' else .65,['dan_silenced']).crop((32,200,640,1028)).save(OUT/'figures'/f'dan_{phase}.png')
    print('Exported',phase,flush=True)
