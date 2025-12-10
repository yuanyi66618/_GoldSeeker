# Gold-Seeker 部署指南

本指南详细介绍Gold-Seeker地球化学找矿预测智能平台的部署方法和最佳实践。

## 📋 目录

- [部署概览](#部署概览)
- [本地部署](#本地部署)
- [服务器部署](#服务器部署)
- [容器化部署](#容器化部署)
- [云平台部署](#云平台部署)
- [集群部署](#集群部署)
- [监控和维护](#监控和维护)
- [安全配置](#安全配置)

## 🌍 部署概览

### 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    负载均衡器                               │
├─────────────────────────────────────────────────────────────┤
│  Web服务器  │  API服务器  │  任务队列  │  缓存服务器    │
├─────────────────────────────────────────────────────────────┤
│  应用服务器集群 (Gold-Seeker实例)                          │
├─────────────────────────────────────────────────────────────┤
│  数据库集群  │  文件存储  │  对象存储  │  备份存储      │
└─────────────────────────────────────────────────────────────┘
```

### 部署类型

1. **单机部署**: 适合开发和小规模使用
2. **服务器部署**: 适合中小规模生产环境
3. **容器化部署**: 适合微服务架构
4. **云平台部署**: 适合大规模弹性部署
5. **集群部署**: 适合高可用和负载均衡

## 💻 本地部署

### 1. 系统要求

#### 最低配置
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 10GB可用空间
- **Python**: 3.9+

#### 推荐配置
- **操作系统**: Windows 11, macOS 12+, Linux (Ubuntu 20.04+)
- **CPU**: 4核心+
- **内存**: 8GB+ RAM
- **存储**: 50GB+ SSD
- **GPU**: 支持CUDA的GPU（可选）

### 2. 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 2. 创建虚拟环境
python -m venv gold-seeker-env
source gold-seeker-env/bin/activate  # Linux/Mac
# 或
gold-seeker-env\Scripts\activate  # Windows

# 3. 安装依赖
pip install -e ".[all]"

# 4. 验证安装
gold-seeker --version
gold-seeker example --dataset synthetic
```

### 3. 配置文件

```yaml
# config/local.yaml
project:
  name: "本地Gold-Seeker实例"
  environment: "development"

data:
  data_dir: "./data"
  cache_dir: "./cache"
  temp_dir: "./temp"

analysis:
  parallel: true
  n_jobs: 4
  chunk_size: 1000

logging:
  level: "INFO"
  file: "./logs/gold_seeker.log"
  console: true

performance:
  memory_limit: "4GB"
  use_gpu: false
```

### 4. 启动服务

```bash
# 启动Web界面
gold-seeker web --host 0.0.0.0 --port 8080

# 启动API服务
gold-seeker api --host 0.0.0.0 --port 8000

# 启动后台任务
gold-seeker worker --n-workers 2
```

## 🖥️ 服务器部署

### 1. 系统准备

#### Ubuntu/Debian

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y gdal-bin libgdal-dev libgeos-dev libproj-dev
sudo apt install -y nginx supervisor redis-server

# 安装Node.js（用于Web界面）
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

#### CentOS/RHEL

```bash
# 安装EPEL仓库
sudo yum install -y epel-release

# 安装系统依赖
sudo yum install -y python3 python3-devel
sudo yum install -y gcc gcc-c++ cmake
sudo yum install -y gdal gdal-devel geos geos-devel proj proj-devel
sudo yum install -y nginx supervisor redis

# 安装Node.js
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs
```

### 2. 应用部署

```bash
# 创建应用用户
sudo useradd -m -s /bin/bash gold-seeker
sudo usermod -aG sudo gold-seeker

# 切换到应用用户
sudo su - gold-seeker

# 克隆代码
git clone https://github.com/your-username/Gold-Seeker.git
cd Gold-Seeker

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e ".[production]"
```

### 3. 配置文件

```yaml
# config/production.yaml
project:
  name: "生产Gold-Seeker实例"
  environment: "production"

data:
  data_dir: "/var/lib/gold-seeker/data"
  cache_dir: "/var/lib/gold-seeker/cache"
  temp_dir: "/tmp/gold-seeker"

database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "gold_seeker"
  user: "gold_seeker"
  password: "${DB_PASSWORD}"

redis:
  host: "localhost"
  port: 6379
  db: 0

analysis:
  parallel: true
  n_jobs: 8
  chunk_size: 5000

logging:
  level: "WARNING"
  file: "/var/log/gold-seeker/gold_seeker.log"
  console: false
  max_size: "100MB"
  backup_count: 10

security:
  secret_key: "${SECRET_KEY}"
  encryption: true
  ssl_required: true

performance:
  memory_limit: "16GB"
  use_gpu: true
  gpu_device: "cuda:0"
```

### 4. 服务配置

#### Nginx配置

```nginx
# /etc/nginx/sites-available/gold-seeker
server {
    listen 80;
    server_name your-domain.com;
    
    # 重定向到HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL配置
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # 静态文件
    location /static/ {
        alias /home/gold-seeker/Gold-Seeker/web/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # API代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Web界面代理
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Supervisor配置

```ini
# /etc/supervisor/conf.d/gold-seeker.conf
[program:gold-seeker-api]
command=/home/gold-seeker/Gold-Seeker/venv/bin/gold-seeker api --host 127.0.0.1 --port 8000
directory=/home/gold-seeker/Gold-Seeker
user=gold-seeker
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/gold-seeker/api.log
environment=PATH="/home/gold-seeker/Gold-Seeker/venv/bin"

[program:gold-seeker-web]
command=/home/gold-seeker/Gold-Seeker/venv/bin/gold-seeker web --host 127.0.0.1 --port 8080
directory=/home/gold-seeker/Gold-Seeker
user=gold-seeker
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/gold-seeker/web.log
environment=PATH="/home/gold-seeker/Gold-Seeker/venv/bin"

[program:gold-seeker-worker]
command=/home/gold-seeker/Gold-Seeker/venv/bin/gold-seeker worker --n-workers 4
directory=/home/gold-seeker/Gold-Seeker
user=gold-seeker
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/gold-seeker/worker.log
environment=PATH="/home/gold-seeker/Gold-Seeker/venv/bin"
```

### 5. 启动服务

```bash
# 创建必要目录
sudo mkdir -p /var/lib/gold-seeker/{data,cache,temp}
sudo mkdir -p /var/log/gold-seeker
sudo chown -R gold-seeker:gold-seeker /var/lib/gold-seeker
sudo chown -R gold-seeker:gold-seeker /var/log/gold-seeker

# 启用Nginx站点
sudo ln -s /etc/nginx/sites-available/gold-seeker /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 启动Supervisor服务
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start gold-seeker:*
```

## 🐳 容器化部署

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV GDAL_CONFIG /usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH /usr/include/gdal
ENV C_INCLUDE_PATH /usr/include/gdal

# 复制依赖文件
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 安装应用
RUN pip install -e .

# 创建非root用户
RUN useradd -m -u 1000 gold-seeker
USER gold-seeker

# 暴露端口
EXPOSE 8000 8080

# 启动命令
CMD ["gold-seeker", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  gold-seeker-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  gold-seeker-web:
    build: .
    command: ["gold-seeker", "web", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  gold-seeker-worker:
    build: .
    command: ["gold-seeker", "worker", "--n-workers", "4"]
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=gold_seeker
      - POSTGRES_USER=gold_seeker
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - gold-seeker-api
      - gold-seeker-web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 3. 部署命令

```bash
# 构建和启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f gold-seeker-api

# 扩展服务
docker-compose up -d --scale gold-seeker-worker=4

# 更新服务
docker-compose pull
docker-compose up -d
```

## ☁️ 云平台部署

### 1. AWS部署

#### ECS部署

```yaml
# aws-ecs-task-definition.json
{
  "family": "gold-seeker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "gold-seeker-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/gold-seeker:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DB_HOST",
          "value": "your-rds-endpoint"
        },
        {
          "name": "REDIS_HOST",
          "value": "your-elasticache-endpoint"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gold-seeker",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Lambda部署

```python
# lambda_handler.py
import json
from gold_seeker import GoldSeeker

# 初始化Gold-Seeker（全局变量，冷启动时初始化）
gs = GoldSeeker()

def lambda_handler(event, context):
    """Lambda处理函数"""
    try:
        # 解析输入
        body = json.loads(event['body'])
        data = body['data']
        target_element = body['target_element']
        
        # 执行分析
        results = gs.quick_analyze(data, target_element)
        
        # 返回结果
        return {
            'statusCode': 200,
            'body': json.dumps(results.to_dict())
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### 2. Azure部署

#### Container Instances

```yaml
# azure-container.yaml
apiVersion: 2019-12-01
location: eastus
name: gold-seeker-group
properties:
  containers:
  - name: gold-seeker-api
    properties:
      image: yourregistry.azurecr.io/gold-seeker:latest
      resources:
        requests:
          cpu: 2.0
          memoryInGb: 4.0
      ports:
      - port: 8000
      environmentVariables:
      - name: DB_HOST
        value: your-database-server.postgres.database.azure.com
      - name: REDIS_HOST
        value: your-redis-cache.redis.cache.windows.net
  osType: Linux
  restartPolicy: Always
type: Microsoft.ContainerInstance/containerGroups
```

### 3. GCP部署

#### Cloud Run

```yaml
# cloudbuild.yaml
steps:
  # 构建Docker镜像
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/gold-seeker:$COMMIT_SHA', '.']
  
  # 推送到Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/gold-seeker:$COMMIT_SHA']
  
  # 部署到Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'gold-seeker'
      - '--image=gcr.io/$PROJECT_ID/gold-seeker:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=4Gi'
      - '--cpu=2'
```

## 🔄 集群部署

### 1. Kubernetes部署

#### Deployment配置

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gold-seeker-api
  labels:
    app: gold-seeker-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gold-seeker-api
  template:
    metadata:
      labels:
        app: gold-seeker-api
    spec:
      containers:
      - name: gold-seeker-api
        image: your-registry/gold-seeker:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: gold-seeker-secrets
              key: db-host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: gold-seeker-secrets
              key: db-password
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service配置

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gold-seeker-api-service
spec:
  selector:
    app: gold-seeker-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress配置

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gold-seeker-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: gold-seeker-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: gold-seeker-api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: gold-seeker-web-service
            port:
              number: 80
```

### 2. Helm Chart

```yaml
# helm/gold-seeker/values.yaml
replicaCount: 3

image:
  repository: your-registry/gold-seeker
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: your-domain.com
      paths:
        - path: /api
          pathType: Prefix
  tls:
    - secretName: gold-seeker-tls
      hosts:
        - your-domain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

database:
  host: postgres
  port: 5432
  name: gold_seeker
  user: gold_seeker
  password: ""

redis:
  host: redis
  port: 6379
  db: 0
```

## 📊 监控和维护

### 1. 监控配置

#### Prometheus配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gold-seeker'
    static_configs:
      - targets: ['gold-seeker-api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### Grafana仪表板

```json
{
  "dashboard": {
    "title": "Gold-Seeker监控",
    "panels": [
      {
        "title": "API请求率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "内存使用",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      }
    ]
  }
}
```

### 2. 日志管理

#### ELK Stack配置

```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "gold-seeker" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "gold-seeker-%{+YYYY.MM.dd}"
  }
}
```

### 3. 备份策略

```bash
#!/bin/bash
# backup.sh

# 设置变量
BACKUP_DIR="/backup/gold-seeker"
DATE=$(date +%Y%m%d_%H%M%S)
DB_BACKUP_FILE="$BACKUP_DIR/db_$DATE.sql"
DATA_BACKUP_FILE="$BACKUP_DIR/data_$DATE.tar.gz"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
pg_dump -h postgres -U gold_seeker gold_seeker > $DB_BACKUP_FILE

# 备份数据文件
tar -czf $DATA_BACKUP_FILE /var/lib/gold-seeker/data

# 上传到云存储
aws s3 cp $DB_BACKUP_FILE s3://gold-seeker-backups/database/
aws s3 cp $DATA_BACKUP_FILE s3://gold-seeker-backups/data/

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "备份完成: $DATE"
```

## 🔒 安全配置

### 1. SSL/TLS配置

```bash
# 生成SSL证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo crontab -e
# 添加以下行
0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. 防火墙配置

```bash
# UFW配置
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. 安全头配置

```nginx
# Nginx安全头
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

## 🚀 性能优化

### 1. 数据库优化

```sql
-- PostgreSQL优化
-- 创建索引
CREATE INDEX idx_geochemical_data_location ON geochemical_data USING GIST (geometry);
CREATE INDEX idx_geochemical_data_au ON geochemical_data (au);

-- 分区表
CREATE TABLE geochemical_data_partitioned (
    LIKE geochemical_data INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- 查询优化
EXPLAIN ANALYZE SELECT * FROM geochemical_data WHERE au > 1.0;
```

### 2. 缓存优化

```python
# Redis缓存配置
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator
```

---

通过遵循本部署指南，您可以成功部署Gold-Seeker平台到各种环境中，从本地开发到大规模生产环境。正确的部署和配置是确保平台稳定运行的关键。