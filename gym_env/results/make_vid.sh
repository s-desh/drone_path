ffmpeg -i "preview_map_frame_%d.png" -s 1200x1200 -r 24 -c:v h264_videotoolbox -q:v 80 -pix_fmt yuv420p -y occ.mp4
ffmpeg -i "frame_%d.png" -s 400x400 -r 24 -c:v h264_videotoolbox -q:v 80 -pix_fmt yuv420p -y drone_2.mp4


ffmpeg -i drone_0.mp4 -i drone_1.mp4 -i drone_2.mp4 -i occ.mp4 \
       -filter_complex "[0:v]scale=400:400[v0];[1:v]scale=400:400[v1];[2:v]scale=400:400[v2];[v0][v1][v2]vstack=inputs=3[left];[3:v]transpose=0, hflip, vflip[right];[left][right]hstack=inputs=2[output]" \
       -c:v h264_videotoolbox -q:v 80 -pix_fmt yuv420p -map "[output]" output.mp4