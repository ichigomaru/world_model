uv run -m lerobot_utils.record \
    --seq_len 50 \
    --height 480 --width 640 \
    --rate 5 \
    --name right \
    --warmup 3 \
    --camera-indices 0 \
    --leader-port /dev/tty.usbmodem57640257221\
    --follower-port /dev/tty.usbmodem58370529971\
    --concatenate \
    --scale 10 \
    --visualize \
    --record

    #./record.sh
    
    #数字で保存
    #nで消す

    #follower-portの次に
    # --concatenate \
    # --scale 10 \

    

    #uv run -m lerobot_utils.concatenator --project_name right  --scale 10.0
    #でデータの統合ができる

    
