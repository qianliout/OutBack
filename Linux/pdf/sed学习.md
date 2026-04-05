## sed 使用参数

    sed [-neifr] [ 命令 ]

## 选项与参数：

*   \-n ：只有经过 sed 特殊处理的那一行（或者命令）才会被列出来。在一般 sed 的用法中，所有来自 STDIN 的数据一般都会被列出到终端上。
*   \-e ：直接在命令列模式上进行 sed 的命令编辑；
*   \-f ：从文件执行 sed 命令，-f filename 则可以运行 filename 内的 sed 命令；
*   \-r ：sed 默认支持正则表达式，使用 -r 开启扩展的正则表达式
*   \-i ：直接修改读取的文件内容，而不是输出到终端。

## 命令说明: \[n1\[,n2]]command

n1, n2 ：在 n1 到 n2 行之间使用命令，举例来说，如果我的命令是需要在 10 到 20 行之间进行的，则 10,20\[ 命令行为 ]

## command：

*   a ：新增， a 的后面可以接字串，而这些字串会在新的一行出现（目前的下一行）～
*   d ：删除
*   i ：插入， i 的后面可以接字串，而这些字串会在新的一行出现（目前的上一行）；
*   p ：查询，亦即将某个选择的数据印出。通常 p 会与参数 sed -n 一起运行～
*   s ：替换，通常这个 s 的命令可以搭配正则
*   \= :只显示行号不显示行内容
*   w 匹配行写入外部文件
*   r 从外部文件写入

## 常用法

*   `10command` 匹配第10行

*   `10,20command ` 匹配从10行开始，到20行结束

*   `10,+5command ` 匹配从10行开始再加5行结束

*   `/patten1/command` 匹配到patten1的行

*   `/patten1/,/pat2en2/command` 匹配到patten1到patten1的行

*   `10，/patten1/command` 从第10行开始匹配到patten1的行

*   `/patten1/,10command` 从patten1到10行结束

*   `sed -n '/\/bin\/bash/s/bash/newbash/g' passwd` 把匹配到/bin/bash的行中的bash转为newbash

*   `sed -n '/\/bin\/bash/s/bash/newbash/g' passwd` 把匹配到/bin/bash的行中的bash转为newbash,注意这里加的转入字符

*   `sed -n '/bash/,/spool/s/bash/newbash/g' passwd` 把匹配到bash,到spool中的行中的bash转换成newbash

## 反向引用

### `&`，`\1`的用法

现在如下文件内容

    hadAAp hello
    word hadBBp
    java hello hadCCp
    goang hadDDp

现在要把hadAAp改为hadAAps，hadBBp改为hadBBps 以此类推

    # 先验证
    sed -n -r 's/had..p/&s/gp' tran.txt
    # 再修改原文件
    sed  -i  's/had..p/&s/g' tran.txt
    sed  -n  -r 's/(had..ps)/\1word/gp' tran.txt
    sed  -n  -r 's/had(..)ps/\1word/gp' tran.txt

`&`，`\1`表示patten匹配到的内容,注意\1时要用（）括起来，`\1`用法更灵活，可以理解成匹配到正则匹配到的部分

在匹配模式中，如查存在变量时，要使用双引号，使用单引号会当做原值处理
在整行范围内替换。如果没有g标记，则只有每行第一个匹配的被替换成。

## 常用修改命令

*   `1s/old/new`
*   `1,10s/old/new/`
*   `1,+4s/old/new/`
*   `/pattern1/s/old/new/`
*   `/pattern1/,/pattern2/s/old/new/`
*   `/pattern1/,10s/old/new/`
*   `10,/pattern1/s/old/new/`

## 外部文件追加

`sed -i '20r /etc/fstab' passwd` 在passwd文件的20行后写入/etc/fstab文件里的内容

`sed -i '/\bin\/bash/r fstab' passwd` 在passwd文件中，匹配bin/bash，在匹配到的行后写入
fstab文件里的内容,是把fstab的内容写入到passwd中

`sed -i '/\/bin\/bash/w fstab' passwd` 把passwd文件中匹配到/bin/bash中的内容写到fstab文件中，这里的hosts文件在当前目录，最好的实践是写绝对路径,这里注意的是是从fstab文件件的第一行开始写，也就是说会覆盖原文件内容
