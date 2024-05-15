#!/usr/bin/expect
# reconstruct.sh [prefix] で実行
set prefix [lindex $argv 0]

set timeout -1

# 必ずspawnで実行
spawn pjsub --interact -g gg18 -L rscgrp=prepost,node=1

# プロンプトを検知したらコマンドを打ち込む
# 適当なコマンドを実行してみる
expect "$ " {send "pwd\r"} 
expect "$ " {send "cd /work/gg18/g18000/falm/app/nedo\r"} 

# reconstructor本体を実行
expect "$ " {send "/work/gg18/g18000/falm/bin/reconstructor $prefix\r"}

# 終了
expect "$ " {send "exit\r"}

# これを忘れなく
interact