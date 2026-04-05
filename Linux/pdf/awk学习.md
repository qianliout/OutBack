## 内置变量

awk 除了 \$ 和数字表示字段还有一些其他的内置变量：

*   `$0 这个表示文本处理时的当前行，$`1 表示文本行被分隔后的第 1 个字段列，\$n 表示文本行被分割后的第 n 个字段列
*   NR 表示文件中的行号，表示当前是第几行
*   NF 表示文件中的当前行被分割的列数，可以理解为 MySQL 数据表里面每一条记录有多少个字段，所以 `$NF 就表示最后一个字段，$`(NF-1) 就表示倒数第二个字段
*   FS 表示 awk 的输入分隔符，默认分隔符为空格和制表符，可以对其进行自定义设置
*   OFS 表示 awk 的输出分隔符，默认为空格，也可以对其进行自定义设置
*   FILENAME 表示当前文件的文件名称，如果同时处理多个文件，它也表示当前文件名称
*   RS 行分隔符，用于分割行，默认为换行符
*   ORS 输出记录的分隔符，默认为换行符

## 插入新字段

在"a b c"在b的后面插入3个字段 e f g

    echo a b c d |awk '{$3 = "e f g "$3}1'
    echo a b c d |awk '{$3 = "e f g "$3;print}'

注意后面有个1，如果没有这个1就不会有输出,最好的方式是第二种，修改值后，会根据OFS重建\$0

### 格式化空白

现在有如下文件

        aaa  bbb           cccc
    bbb cccc       ddd    

`awk '{$1=$1;print}' 1.txt`
重新赋值就行，注意的是，这里不能用  `$0=$`0,这样是没有变动的

## 筛选IPv4地址

从ifconfig的结果中筛选不是lo网卡的IPv4地址

```
ifconfig |awk '/inet / && !($2 ~ /^127/) {print $2}'

# 按段落读取
ifconfig |awk 'BEGIN{RS=""}!/^lo:/{print $6}'

ifconfig |awk 'BEGIN{RS="";FS="\n"}!/^lo:/{FS=" ";$0=$0;print $6}'

```

*   RS=""  就会按段落读取（默认是按行读取）
*   RS="\0"  一次性读取所有数据，但是不安全，因为文件中可能有"\0"的字符
*   RS="^\$"  直正的一次性读取所有的数据
*   RS="\n+"  按行读取，但是忽略空行

## 读取配置文件中的内容

getline的返回值：>0 表示读取到数据, =0 表示数据已读完(EOF), <0 表示读取出错，可能是权限等问题

如下init文件（hello.ini）

    [base]
    name=os
    age=14
    [mango]
    name=mangguo
    age=5

现在想读取base下的配置文件

如下awk文件

    index($0,"[base]"){
        while ((getline var )>0){
            if (var ~  /\[.*\]/ ){
                exit
            }
            print var
        }
    }

然后执行
`awk -f a.awk hello.ini`

## 根据字段去重

有如下数据,根据uid去重，输出第一次出现的行

    2019-01-13_12:00_index?uid=123
    2019-01-13_14:00_index?uid=124
    2019-03-13_12:00_index?uid=125
    2019-01-14_12:00_index?uid=123

`awk -F "?" '{arr[$2]++;if(arr[$2]==1){print}}' log.txt`

## 统计单词个数

有如下数据

    hello
    word
    hello
    word
    mysql
    golang
    java
    java
    hello



    awk '{arr[$1]++}END{for (i in arr) print i,arr[i]}' words.txt 

    awk '{arr[$1]++}END{for (i in arr) {print i,arr[i]}}' words.txt

## 统计tcp连接的数量

    netstat -tnap |awk '{arr[$6]++}END {for (i in arr){print i,arr[i]}}'

    netstat -tnap |awk '/^tcp/ {arr[$6]++}END {for (i in arr){print i,arr[i]}}'

    # 排序加去重
    netstat -tnap |awk '/^tcp/ {arr[$6]++}END {for (i in arr){print i,arr[i]}}' |sort -nrk2 |uniq -c

## 在awk中排序

如下文件内容

    192.168.1.120 www.baidu.com 200
    192.168.1.120 www.baidu.com 302
    192.168.1.120 www.baidu.com 400
    192.168.1.122 www.baidu.com 403
    192.168.1.123 www.baidu.com 500

找出状态码不是200的ip的个数，并从大到小排序

    awk '$3!=200{arr[$1]++}END{for (i in arr){print i , arr[i]}}' user.log  |sort -nrk2

    awk '$3!=200{arr[$1]++}END{PROCINFO["sorted_in"]="@val_num_desc" ;for (i in arr){print i , arr[i]}}' user.log

## 处理确失的字段

有如下数据

    Name   Age Email        QQ
    hello      112@163.com  123232
    word   123              123232

缺少了部分字段，如果再以空格分隔就不行

    awk '{BEGIN{FIELDWIDTHS="5 1:3 1:11 1:6"}}' student.txt

表示第一行取5个字符，第二行是先跳过1个字符再取3个字符以此类推，这个命令只有在gawk中才可以使用，默认未安装gawk
