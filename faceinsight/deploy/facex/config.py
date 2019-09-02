# vi: set ft=python sts=4 ts=4 sw=4 et:

FACTORS = {'A': '乐群性*',
           'B': '聪慧性',
           'C': '稳定性',
           'E': '恃强性',
           'F': '兴奋性*',
           'G': '有恒性',
           'H': '敢为性*',
           'I': '敏感性*',
           'L': '怀疑性*',
           'M': '幻想性',
           'N': '世故性*',
           'O': '忧虑性',
           'Q1': '实验性',
           'Q2': '独立性*',
           'Q3': '自律性*',
           'Q4': '紧张性'}

ROOT_DIR = '/home/huanglj/repo/FaceInsight/faceinsight'
#ROOT_DIR = '/Users/sealhuang/repo/FaceInsight/faceinsight'
BASEPATH = '/tmp'
UPLOAD_FOLDER = './static/uploads'
ALLOW_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
print('BASEPATH', BASEPATH)
print('UPLOAD_FOLDER', UPLOAD_FOLDER)

#APP_URL = '127.0.0.1'
APP_URL = '192.168.1.10'
DET_URL = '127.0.0.1'
INF_URL = '127.0.0.1'
