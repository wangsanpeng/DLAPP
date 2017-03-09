#!/usr/bin/env python
# coding=utf-8
from pullword import pullword

s = pullword(u"习近平指出，中柬两国人民传统友谊源远流长。中柬友好是由中国老一辈领导人和西哈努克太皇共同缔造和精心培育的，弥足珍贵。进入新的历史时期，中柬关系又增添新的活力，得到长足发展。去年我同西哈莫尼国王成功互访。当前，两国政治上高度互信，经济上互利合作，在实现国家发展中互帮互助，在国际和地区事务中密切配合。柬埔寨王室长期以来积极致力于中柬友好事业，为两国关系发展作出了重要贡献。我们愿同柬方携手努力，推动中柬全面战略合作不断迈上新台阶，更好造福两国人民。")
print s
with open('/Users/cheng/Downloads/xxxx', 'w') as f:
    f.writelines([ ss[0]+ss[1]+' ' for ss in s])

