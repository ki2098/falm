#!/usr/bin/expect
# reconstruct.sh [prefix] で実行
set prefix [lindex $argv 0]

# reconstructorは少し時間がかかるので、タイムアウトを無効化しないと、reconstructor後のコマンドは実行されなくなる
set timeout -1

# 必ずspawnで実行
spawn pjsub --interact -g gg18 -L rscgrp=prepost,node=1

# プロンプトを検知したらコマンドを打ち込む
# 適当なコマンドを実行してみる
expect "$ " {send "pwd\r"}

# データは一応/work/gg18/g18000/falm/app/nedo/dataに保存してある
expect "$ " {send "cd /work/gg18/g18000/falm/app/nedo\r"} 

# reconstructor本体を実行
expect "$ " {send "/work/gg18/g18000/falm/bin/reconstructor $prefix\r"}

# 終了
expect "$ " {send "exit\r"}

# これを忘れなく
interact