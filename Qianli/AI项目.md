
1：选者一个大模型
2：使用 lora 的方式微调 漏洞信息
3：导出我们漏洞库中的所有数据进行微调

mysqldump -u root -pMysql-ha@123 --single-transaction --no-create-info --skip-triggers --compact my_database my_large_table > large_table_data.sql

mysqldump --host=172.21.2.229 --port=30036 -uroot -pMysql-ha@123 --databases ivan --tables ivan_scan_image_vuln > vuln.sql