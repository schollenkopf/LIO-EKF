docker network create ros
docker run -d --rm --net=ros \
   --env="DISPLAY_WIDTH=2000" --env="DISPLAY_HEIGHT=1200" --env="RUN_XTERM=no" \
   --name=novnc -p=8080:8080 \
   theasp/novnc:latest