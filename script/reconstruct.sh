#!/usr/bin/expect

#必ずspawnで実行
spawn pjsub --interact -g gg18 -L rscgrp=prepost,node=1

#適当なコマンドを実行してみる
expect "$ " {send "pwd\r"} 

expect "$ " {send "cd /work/gg18/g18000/falm/app/nedo\r"} 

#「data\uvpw1.0」の部分は実際のデータパスに変更
expect "$ " {send "/work/gg18/g18000/falm/bin/reconstructor $1\r"}

#終了
expect "$ " {send "exit\r"}

#これを忘れなく
interact