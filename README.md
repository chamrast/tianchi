- 组队信息：https://shimo.im/sheets/QK1P8vycscw7lsL6/MODOC/
- 学员手册：http://datawhale.club/t/topic/1421
- github：https://github.com/datawhalechina/team-learning
- 论坛：http://datawhale.club/
- 比赛地址：https://tianchi.aliyun.com/competition/entrance/531872/introduction
- B 站 Datawhale：https://space.bilibili.com/431850986?from=search&seid=724423084384707923

# 模式

1. 学习模式：通过组队的形式共同学习，互相交流，互相督促，共同进步，组队的人数为【5-10人】；
2. 打卡规则：每次的task大家要将学习内容整理成博客发布在CSDN、Github或其他平台。学习内容包括但不限于：学习到的知识总结，学习遇到的问题和对本次task的感受，编辑格式不限。
3. 每次打卡之前会适时为大家提供打卡链接。每位同学需要在打卡截止时间前完成打卡，否则会被【抱出群】哦。

# DSW

在 DSW 内新建一个 notebook，就会在天池实验室「我公开的项目」那里显示，即一个 notebook 一个项目？

默认目录是 /home/admin/jupyter/，是持久化目录，只有 5G。

- 解压数据之后，保存文件报错，将 *__MACOSX* 删了之下又可以了，所以是磁盘不够的问题？

所以每次将数据下去其它地方？有得挂载最好。

# 环境

- [conda修改环境/包的安装路径+镜像源更新](https://zhuanlan.zhihu.com/p/337080930)
- [更改 Python 的 pip install 默认使用的pip库以及默认安装路径](https://blog.csdn.net/C_chuxin/article/details/82962797)
- [用conda创建python虚拟环境](https://blog.csdn.net/lyy14011305/article/details/59500819)
- [如何在Jupyter Notebook中使用Python虚拟环境？](https://www.jianshu.com/p/afea092dda1d)，没权限装 nb_conda，这个方法不行。
- [在jupyter notebook上使用python虚拟环境](https://www.jianshu.com/p/f70ea020e6f9)

~~以上基本无用，因为 /home/admin/ **不**是持久化目录，改 conda 路径重启就会没。~~

原有 pytorch 版本似乎低，可以新建虚拟环境，用新 pytorch。先执行 `source activate`，再执行 `conda activate xxx` 换虚拟环境。

# 视频

- task 2：https://tianchi.aliyun.com/course/live/1613
- task 3：https://tianchi.aliyun.com/course/live/1614