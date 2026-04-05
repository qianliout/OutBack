ssh 登录时需要手动输入

```shell
#!/usr/bin/expect 

spawn ssh sit

expect "Code" {
    set code [exec oathtool --totp -b IVU5RW7IPIWJ7QXN ]
    puts "code is $code"
} 
send "$code\r"

interactsh
```

