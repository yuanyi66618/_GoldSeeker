# Gold-Seeker 安全指南

本指南详细介绍Gold-Seeker地球化学找矿预测智能平台的安全架构、最佳实践和合规要求。

## 📋 目录

- [安全概览](#安全概览)
- [威胁模型](#威胁模型)
- [安全架构](#安全架构)
- [数据安全](#数据安全)
- [访问控制](#访问控制)
- [网络安全](#网络安全)
- [加密保护](#加密保护)
- [安全监控](#安全监控)
- [合规要求](#合规要求)
- [安全最佳实践](#安全最佳实践)

## 🛡️ 安全概览

### 安全目标

Gold-Seeker平台的安全设计遵循以下核心目标：

1. **保密性 (Confidentiality)**: 确保敏感数据不被未授权访问
2. **完整性 (Integrity)**: 保证数据在传输和存储过程中不被篡改
3. **可用性 (Availability)**: 确保系统持续可用，防止拒绝服务攻击
4. **可追溯性 (Accountability)**: 记录所有操作，支持审计和追溯
5. **隐私保护 (Privacy)**: 保护个人隐私和敏感信息

### 安全等级

| 安全等级 | 描述 | 适用场景 | 保护措施 |
|----------|------|----------|----------|
| Level 1 | 基础安全 | 开发环境、测试数据 | 基本认证、数据加密 |
| Level 2 | 标准安全 | 生产环境、商业数据 | 多因素认证、访问控制 |
| Level 3 | 高级安全 | 企业环境、敏感数据 | 端到端加密、审计日志 |
| Level 4 | 军事安全 | 政府项目、机密数据 | 零信任架构、物理隔离 |

### 合规框架

- **ISO 27001**: 信息安全管理体系
- **GDPR**: 通用数据保护条例
- **SOC 2**: 服务组织控制
- **NIST**: 网络安全框架
- **等保2.0**: 网络安全等级保护

## 🎯 威胁模型

### 威胁分类

#### 外部威胁

1. **网络攻击**
   - DDoS攻击
   - SQL注入
   - 跨站脚本(XSS)
   - 中间人攻击

2. **恶意软件**
   - 病毒、木马
   - 勒索软件
   - 间谍软件
   - 挖矿软件

3. **社会工程**
   - 钓鱼攻击
   - 身份冒充
   - 恶意链接
   - 电话诈骗

#### 内部威胁

1. **恶意内部人员**
   - 数据窃取
   - 系统破坏
   - 权限滥用
   - 信息泄露

2. **无意内部威胁**
   - 操作失误
   - 配置错误
   - 密码泄露
   - 设备丢失

#### 系统威胁

1. **软件漏洞**
   - 代码缺陷
   - 依赖漏洞
   - 配置错误
   - 逻辑漏洞

2. **硬件故障**
   - 磁盘损坏
   - 网络中断
   - 电源故障
   - 自然灾害

### 风险评估矩阵

| 威胁类型 | 可能性 | 影响程度 | 风险等级 | 缓解措施 |
|----------|--------|----------|----------|----------|
| 数据泄露 | 中 | 高 | 高 | 加密、访问控制 |
| 系统入侵 | 中 | 高 | 高 | 防火墙、入侵检测 |
| 内部威胁 | 低 | 高 | 中 | 背景调查、权限管理 |
| 设备丢失 | 中 | 中 | 中 | 设备加密、远程擦除 |
| 自然灾害 | 低 | 高 | 中 | 备份、灾难恢复 |

## 🏗️ 安全架构

### 零信任架构

```
┌─────────────────────────────────────────────────────────────┐
│                    安全控制层                                │
├─────────────────────────────────────────────────────────────┤
│  身份认证  │  设备信任  │  网络分段  │  应用安全  │  数据保护  │
├─────────────────────────────────────────────────────────────┤
│                    微服务架构                                │
├─────────────────────────────────────────────────────────────┤
│  API网关  │  认证服务  │  授权服务  │  审计服务  │  监控服务  │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层                                │
├─────────────────────────────────────────────────────────────┤
│  容器安全  │  网络安全  │  存储安全  │  密钥管理  │  备份恢复  │
└─────────────────────────────────────────────────────────────┘
```

### 安全组件

#### 1. 身份认证系统

```python
# 认证服务实现
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class AuthenticationService:
    """身份认证服务"""
    
    def __init__(self, secret_key: str, token_expiry: int = 3600):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.user_store = UserStore()
        self.session_store = SessionStore()
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        user = self.user_store.get_user(username)
        if not user:
            return None
        
        # 验证密码
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            return None
        
        # 生成JWT令牌
        token = self.generate_token(user)
        
        # 记录会话
        self.session_store.create_session(user.id, token)
        
        return token
    
    def generate_token(self, user: User) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'roles': user.roles,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # 检查会话是否有效
            if not self.session_store.is_session_valid(payload['user_id'], token):
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def logout(self, token: str) -> bool:
        """用户登出"""
        payload = self.verify_token(token)
        if payload:
            self.session_store.invalidate_session(payload['user_id'], token)
            return True
        return False

# 多因素认证
class MFAService:
    """多因素认证服务"""
    
    def __init__(self):
        self.totp_service = TOTPService()
        self.sms_service = SMSService()
        self.email_service = EmailService()
    
    def send_otp(self, user: User, method: str = 'totp') -> bool:
        """发送一次性密码"""
        if method == 'totp':
            return self.totp_service.generate_secret(user)
        elif method == 'sms':
            return self.sms_service.send_otp(user.phone)
        elif method == 'email':
            return self.email_service.send_otp(user.email)
        return False
    
    def verify_otp(self, user: User, otp: str, method: str = 'totp') -> bool:
        """验证一次性密码"""
        if method == 'totp':
            return self.totp_service.verify_otp(user, otp)
        elif method == 'sms':
            return self.sms_service.verify_otp(user, otp)
        elif method == 'email':
            return self.email_service.verify_otp(user, otp)
        return False
```

#### 2. 授权控制系统

```python
# 基于角色的访问控制(RBAC)
from enum import Enum
from typing import List, Set

class Permission(Enum):
    """权限枚举"""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    ANALYZE_DATA = "analyze_data"
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    SYSTEM_CONFIG = "system_config"

class Role(Enum):
    """角色枚举"""
    VIEWER = "viewer"
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMIN = "admin"

class RBACService:
    """基于角色的访问控制服务"""
    
    def __init__(self):
        self.role_permissions = {
            Role.VIEWER: {
                Permission.READ_DATA
            },
            Role.ANALYST: {
                Permission.READ_DATA,
                Permission.ANALYZE_DATA,
                Permission.WRITE_DATA
            },
            Role.MANAGER: {
                Permission.READ_DATA,
                Permission.ANALYZE_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.VIEW_LOGS
            },
            Role.ADMIN: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.ANALYZE_DATA,
                Permission.MANAGE_USERS,
                Permission.VIEW_LOGS,
                Permission.SYSTEM_CONFIG
            }
        }
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """检查用户权限"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """获取用户权限"""
        permissions = set()
        
        for role in user.roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        
        return permissions
    
    def add_role_to_user(self, user: User, role: Role) -> bool:
        """为用户添加角色"""
        if role not in user.roles:
            user.roles.append(role)
            return True
        return False
    
    def remove_role_from_user(self, user: User, role: Role) -> bool:
        """从用户移除角色"""
        if role in user.roles:
            user.roles.remove(role)
            return True
        return False

# 权限装饰器
def require_permission(permission: Permission):
    """权限检查装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取当前用户
            current_user = get_current_user()
            
            # 检查权限
            rbac = RBACService()
            if not rbac.has_permission(current_user, permission):
                raise PermissionError(f"需要权限: {permission.value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@require_permission(Permission.ANALYZE_DATA)
def analyze_geochemical_data(data):
    """分析地球化学数据"""
    return perform_analysis(data)
```

#### 3. 审计日志系统

```python
# 审计日志服务
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class AuditLogger:
    """审计日志服务"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, 
                  event_type: str,
                  user_id: str,
                  resource: str,
                  action: str,
                  result: str,
                  details: Optional[Dict[str, Any]] = None):
        """记录审计事件"""
        audit_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'result': result,
            'ip_address': get_client_ip(),
            'user_agent': get_user_agent(),
            'details': details or {}
        }
        
        self.logger.info(json.dumps(audit_record))
    
    def log_login(self, user_id: str, success: bool, ip_address: str):
        """记录登录事件"""
        self.log_event(
            event_type="LOGIN",
            user_id=user_id,
            resource="AUTH",
            action="LOGIN",
            result="SUCCESS" if success else "FAILED",
            details={'ip_address': ip_address}
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str):
        """记录数据访问事件"""
        self.log_event(
            event_type="DATA_ACCESS",
            user_id=user_id,
            resource=resource,
            action=action,
            result="SUCCESS"
        )
    
    def log_system_change(self, user_id: str, component: str, change: str):
        """记录系统变更事件"""
        self.log_event(
            event_type="SYSTEM_CHANGE",
            user_id=user_id,
            resource=component,
            action="MODIFY",
            result="SUCCESS",
            details={'change': change}
        )

# 审计装饰器
def audit_action(action: str, resource: str):
    """审计装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_user = get_current_user()
            audit_logger = AuditLogger()
            
            try:
                result = func(*args, **kwargs)
                audit_logger.log_event(
                    event_type="FUNCTION_CALL",
                    user_id=current_user.id,
                    resource=resource,
                    action=action,
                    result="SUCCESS",
                    details={'function': func.__name__}
                )
                return result
            except Exception as e:
                audit_logger.log_event(
                    event_type="FUNCTION_CALL",
                    user_id=current_user.id,
                    resource=resource,
                    action=action,
                    result="FAILED",
                    details={'function': func.__name__, 'error': str(e)}
                )
                raise
        return wrapper
    return decorator
```

## 🔐 数据安全

### 数据分类

| 数据类别 | 描述 | 安全等级 | 保护措施 |
|----------|------|----------|----------|
| 公开数据 | 公开的地质数据 | Level 1 | 基本保护 |
| 内部数据 | 内部使用的数据 | Level 2 | 访问控制 |
| 敏感数据 | 商业敏感数据 | Level 3 | 加密存储 |
| 机密数据 | 国家机密数据 | Level 4 | 端到端加密 |

### 数据加密

#### 静态数据加密

```python
# 数据加密服务
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryptionService:
    """数据加密服务"""
    
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self) -> bytes:
        """派生加密密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key
    
    def encrypt_data(self, data: bytes) -> bytes:
        """加密数据"""
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """解密数据"""
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: str, output_path: str) -> bool:
        """加密文件"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.encrypt_data(data)
            
            with open(output_path, 'wb') as f:
                f.write(self.salt + encrypted_data)
            
            return True
        except Exception as e:
            print(f"文件加密失败: {e}")
            return False
    
    def decrypt_file(self, file_path: str, output_path: str) -> bool:
        """解密文件"""
        try:
            with open(file_path, 'rb') as f:
                salt = f.read(16)
                encrypted_data = f.read()
            
            # 重新创建解密器
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password))
            cipher = Fernet(key)
            
            decrypted_data = cipher.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            return True
        except Exception as e:
            print(f"文件解密失败: {e}")
            return False

# 数据库字段加密
class DatabaseEncryption:
    """数据库字段加密"""
    
    def __init__(self, encryption_service: DataEncryptionService):
        self.encryption_service = encryption_service
    
    def encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """加密敏感字段"""
        sensitive_fields = ['phone', 'email', 'address', 'ssn']
        
        encrypted_data = data.copy()
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_value = self.encryption_service.encrypt_data(
                    encrypted_data[field].encode()
                )
                encrypted_data[field] = encrypted_value.decode()
        
        return encrypted_data
    
    def decrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解密敏感字段"""
        sensitive_fields = ['phone', 'email', 'address', 'ssn']
        
        decrypted_data = data.copy()
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_value = self.encryption_service.decrypt_data(
                    decrypted_data[field].encode()
                )
                decrypted_data[field] = decrypted_value.decode()
        
        return decrypted_data
```

#### 传输数据加密

```python
# TLS/SSL配置
from flask import Flask
from flask_sslify import SSLify

def create_secure_app():
    """创建安全的应用"""
    app = Flask(__name__)
    
    # 强制HTTPS
    sslify = SSLify(app)
    
    # 配置安全头
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response
    
    return app

# API安全
from flask_httpauth import HTTPTokenAuth

token_auth = HTTPTokenAuth(scheme='Bearer')

@token_auth.verify_token
def verify_token(token):
    """验证API令牌"""
    auth_service = AuthenticationService()
    payload = auth_service.verify_token(token)
    return payload is not None

@app.route('/api/secure-data')
@token_auth.login_required
def get_secure_data():
    """获取安全数据"""
    return {'data': 'secure information'}
```

### 数据脱敏

```python
# 数据脱敏服务
import re
import hashlib
from typing import Any, Dict

class DataMaskingService:
    """数据脱敏服务"""
    
    def __init__(self):
        self.masking_rules = {
            'phone': self._mask_phone,
            'email': self._mask_email,
            'id_card': self._mask_id_card,
            'address': self._mask_address,
            'name': self._mask_name
        }
    
    def mask_data(self, data: Dict[str, Any], rules: Dict[str, str] = None) -> Dict[str, Any]:
        """脱敏数据"""
        if rules is None:
            rules = {}
        
        masked_data = data.copy()
        
        for field, rule in rules.items():
            if field in masked_data:
                if rule in self.masking_rules:
                    masked_data[field] = self.masking_rules[rule](masked_data[field])
                else:
                    masked_data[field] = self._mask_default(masked_data[field])
        
        return masked_data
    
    def _mask_phone(self, phone: str) -> str:
        """脱敏手机号"""
        if len(phone) >= 11:
            return phone[:3] + '****' + phone[-4:]
        return '****'
    
    def _mask_email(self, email: str) -> str:
        """脱敏邮箱"""
        if '@' in email:
            local, domain = email.split('@', 1)
            if len(local) > 2:
                masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
            else:
                masked_local = '*' * len(local)
            return masked_local + '@' + domain
        return '****@****.com'
    
    def _mask_id_card(self, id_card: str) -> str:
        """脱敏身份证号"""
        if len(id_card) >= 18:
            return id_card[:6] + '********' + id_card[-4:]
        return '********************'
    
    def _mask_address(self, address: str) -> str:
        """脱敏地址"""
        if len(address) > 10:
            return address[:6] + '****'
        return '****'
    
    def _mask_name(self, name: str) -> str:
        """脱敏姓名"""
        if len(name) >= 2:
            return name[0] + '*' * (len(name) - 1)
        return '*'
    
    def _mask_default(self, value: str) -> str:
        """默认脱敏"""
        if len(value) > 4:
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
        return '*' * len(value)
    
    def hash_sensitive_data(self, data: str) -> str:
        """哈希敏感数据"""
        return hashlib.sha256(data.encode()).hexdigest()
```

## 🌐 网络安全

### 防火墙配置

```yaml
# iptables规则
- name: Configure firewall
  iptables:
    chain: INPUT
    protocol: tcp
    destination_port: "{{ item }}"
    jump: ACCEPT
  with_items:
    - 22    # SSH
    - 80    # HTTP
    - 443   # HTTPS
    - 8000  # API
    - 8080  # Web界面

- name: Drop all other traffic
  iptables:
    chain: INPUT
    policy: DROP
```

### 入侵检测系统

```python
# 入侵检测服务
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

class IntrusionDetectionSystem:
    """入侵检测系统"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'sql_injection': [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"drop\s+table",
                r"insert\s+into"
            ],
            'xss': [
                r"<script",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*="
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ]
        }
        
        self.ip_requests = defaultdict(list)
        self.blocked_ips = set()
        self.rate_limit = 100  # 每分钟请求数
        self.block_duration = 3600  # 封禁时间(秒)
    
    def analyze_request(self, ip: str, user_agent: str, request_data: str) -> Dict[str, Any]:
        """分析请求"""
        current_time = time.time()
        
        # 检查IP是否被封禁
        if ip in self.blocked_ips:
            return {'status': 'blocked', 'reason': 'IP blocked'}
        
        # 检查请求频率
        self.ip_requests[ip].append(current_time)
        self.ip_requests[ip] = [
            req_time for req_time in self.ip_requests[ip]
            if current_time - req_time < 60
        ]
        
        if len(self.ip_requests[ip]) > self.rate_limit:
            self.block_ip(ip)
            return {'status': 'blocked', 'reason': 'Rate limit exceeded'}
        
        # 检查恶意模式
        for attack_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_data, re.IGNORECASE):
                    self.log_suspicious_activity(ip, attack_type, request_data)
                    return {
                        'status': 'suspicious',
                        'reason': f'{attack_type} detected',
                        'pattern': pattern
                    }
        
        return {'status': 'safe'}
    
    def block_ip(self, ip: str):
        """封禁IP"""
        self.blocked_ips.add(ip)
        self.log_security_event('IP_BLOCKED', {'ip': ip})
    
    def log_suspicious_activity(self, ip: str, attack_type: str, request_data: str):
        """记录可疑活动"""
        self.log_security_event('SUSPICIOUS_ACTIVITY', {
            'ip': ip,
            'attack_type': attack_type,
            'request_data': request_data[:1000]  # 限制长度
        })
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """记录安全事件"""
        audit_logger = AuditLogger()
        audit_logger.log_event(
            event_type=event_type,
            user_id="SYSTEM",
            resource="SECURITY",
            action="DETECT",
            result="DETECTED",
            details=details
        )
```

### DDoS防护

```python
# DDoS防护服务
import time
from collections import defaultdict, deque

class DDoSProtectionService:
    """DDoS防护服务"""
    
    def __init__(self):
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.rate_limits = {
            'global': 10000,    # 全局每秒请求数
            'per_ip': 100,      # 每IP每秒请求数
            'per_user': 50      # 每用户每秒请求数
        }
        self.blocked_ips = set()
        self.blocked_users = set()
    
    def check_request(self, ip: str, user_id: str = None) -> bool:
        """检查请求是否允许"""
        current_time = time.time()
        
        # 清理过期记录
        self._cleanup_expired_requests(current_time)
        
        # 检查全局限制
        global_requests = sum(len(requests) for requests in self.request_history.values())
        if global_requests > self.rate_limits['global']:
            return False
        
        # 检查IP限制
        ip_requests = len(self.request_history[ip])
        if ip_requests > self.rate_limits['per_ip']:
            self.blocked_ips.add(ip)
            return False
        
        # 检查用户限制
        if user_id:
            user_requests = sum(
                1 for requests in self.request_history.values()
                if any(req[1] == user_id for req in requests)
            )
            if user_requests > self.rate_limits['per_user']:
                self.blocked_users.add(user_id)
                return False
        
        # 记录请求
        self.request_history[ip].append((current_time, user_id))
        
        return True
    
    def _cleanup_expired_requests(self, current_time: float):
        """清理过期请求记录"""
        for ip in list(self.request_history.keys()):
            requests = self.request_history[ip]
            while requests and current_time - requests[0][0] > 1.0:
                requests.popleft()
            
            if not requests:
                del self.request_history[ip]
    
    def is_ip_blocked(self, ip: str) -> bool:
        """检查IP是否被封禁"""
        return ip in self.blocked_ips
    
    def is_user_blocked(self, user_id: str) -> bool:
        """检查用户是否被封禁"""
        return user_id in self.blocked_users
    
    def unblock_ip(self, ip: str):
        """解封IP"""
        self.blocked_ips.discard(ip)
    
    def unblock_user(self, user_id: str):
        """解封用户"""
        self.blocked_users.discard(user_id)
```

## 🔑 密钥管理

### 密钥管理系统

```python
# 密钥管理服务
import os
import json
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class KeyManagementService:
    """密钥管理服务"""
    
    def __init__(self, master_key_path: str = "master.key"):
        self.master_key_path = master_key_path
        self.master_key = self._load_or_generate_master_key()
        self.key_store = {}
    
    def _load_or_generate_master_key(self) -> bytes:
        """加载或生成主密钥"""
        if os.path.exists(self.master_key_path):
            with open(self.master_key_path, 'rb') as f:
                return f.read()
        else:
            master_key = secrets.token_bytes(32)
            with open(self.master_key_path, 'wb') as f:
                f.write(master_key)
            os.chmod(self.master_key_path, 0o600)  # 仅所有者可读写
            return master_key
    
    def derive_key(self, context: str, length: int = 32) -> bytes:
        """派生密钥"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=context.encode(),
            backend=default_backend()
        )
        return hkdf.derive(self.master_key)
    
    def generate_data_key(self, key_id: str) -> bytes:
        """生成数据密钥"""
        data_key = secrets.token_bytes(32)
        encrypted_key = self._encrypt_key(data_key, key_id)
        
        self.key_store[key_id] = {
            'encrypted_key': encrypted_key,
            'created_at': time.time()
        }
        
        return data_key
    
    def _encrypt_key(self, key: bytes, context: str) -> bytes:
        """加密密钥"""
        iv = secrets.token_bytes(16)
        cipher_key = self.derive_key(f"key_encryption_{context}")
        
        cipher = Cipher(
            algorithms.AES(cipher_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        padded_key = self._pad_data(key)
        encrypted_key = encryptor.update(padded_key) + encryptor.finalize()
        
        return iv + encrypted_key
    
    def _decrypt_key(self, encrypted_key: bytes, context: str) -> bytes:
        """解密密钥"""
        iv = encrypted_key[:16]
        ciphertext = encrypted_key[16:]
        
        cipher_key = self.derive_key(f"key_encryption_{context}")
        
        cipher = Cipher(
            algorithms.AES(cipher_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        padded_key = decryptor.update(ciphertext) + decryptor.finalize()
        
        return self._unpad_data(padded_key)
    
    def _pad_data(self, data: bytes) -> bytes:
        """PKCS7填充"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """移除PKCS7填充"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def rotate_keys(self):
        """轮换密钥"""
        new_master_key = secrets.token_bytes(32)
        
        # 重新加密所有存储的密钥
        for key_id, key_info in self.key_store.items():
            old_encrypted_key = key_info['encrypted_key']
            old_key = self._decrypt_key(old_encrypted_key, key_id)
            
            # 临时切换到新主密钥
            old_master_key = self.master_key
            self.master_key = new_master_key
            
            new_encrypted_key = self._encrypt_key(old_key, key_id)
            
            # 恢复旧主密钥
            self.master_key = old_master_key
            
            self.key_store[key_id]['encrypted_key'] = new_encrypted_key
        
        # 更新主密钥
        self.master_key = new_master_key
        with open(self.master_key_path, 'wb') as f:
            f.write(new_master_key)
```

## 📊 安全监控

### 安全事件监控

```python
# 安全监控服务
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

class SecurityMonitoringService:
    """安全监控服务"""
    
    def __init__(self):
        self.security_events = []
        self.alert_thresholds = {
            'failed_logins': 5,        # 失败登录次数
            'suspicious_requests': 10,  # 可疑请求数
            'data_access_anomaly': 3,   # 数据访问异常
            'system_errors': 20        # 系统错误数
        }
        self.alert_handlers = []
    
    def add_security_event(self, event_type: str, details: Dict[str, Any]):
        """添加安全事件"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        self._check_alert_conditions(event)
    
    def _check_alert_conditions(self, event: Dict[str, Any]):
        """检查告警条件"""
        current_time = datetime.utcnow()
        
        # 检查失败登录
        if event['event_type'] == 'FAILED_LOGIN':
            recent_failures = self._count_recent_events(
                'FAILED_LOGIN', 
                timedelta(minutes=15)
            )
            if recent_failures >= self.alert_thresholds['failed_logins']:
                self._trigger_alert('BRUTE_FORCE_ATTACK', {
                    'failed_attempts': recent_failures,
                    'time_window': '15 minutes'
                })
        
        # 检查可疑请求
        if event['event_type'] == 'SUSPICIOUS_REQUEST':
            recent_suspicious = self._count_recent_events(
                'SUSPICIOUS_REQUEST',
                timedelta(minutes=5)
            )
            if recent_suspicious >= self.alert_thresholds['suspicious_requests']:
                self._trigger_alert('POTENTIAL_ATTACK', {
                    'suspicious_requests': recent_suspicious,
                    'time_window': '5 minutes'
                })
        
        # 检查数据访问异常
        if event['event_type'] == 'DATA_ACCESS':
            recent_access = self._count_recent_events(
                'DATA_ACCESS',
                timedelta(minutes=10)
            )
            if recent_access >= self.alert_thresholds['data_access_anomaly']:
                self._trigger_alert('DATA_ACCESS_ANOMALY', {
                    'access_count': recent_access,
                    'time_window': '10 minutes'
                })
    
    def _count_recent_events(self, event_type: str, time_window: timedelta) -> int:
        """计算最近事件数量"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - time_window
        
        count = 0
        for event in self.security_events:
            if (event['event_type'] == event_type and
                datetime.fromisoformat(event['timestamp']) >= cutoff_time):
                count += 1
        
        return count
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """触发告警"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert_type': alert_type,
            'details': details,
            'severity': self._get_alert_severity(alert_type)
        }
        
        # 通知所有告警处理器
        for handler in self.alert_handlers:
            handler.handle_alert(alert)
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """获取告警严重程度"""
        severity_map = {
            'BRUTE_FORCE_ATTACK': 'HIGH',
            'POTENTIAL_ATTACK': 'MEDIUM',
            'DATA_ACCESS_ANOMALY': 'MEDIUM',
            'SYSTEM_ERROR': 'LOW'
        }
        return severity_map.get(alert_type, 'LOW')
    
    def add_alert_handler(self, handler):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def get_security_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """获取安全摘要"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - time_window
        
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event['timestamp']) >= cutoff_time
        ]
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event['event_type']] += 1
        
        return {
            'time_window': str(time_window),
            'total_events': len(recent_events),
            'event_counts': dict(event_counts),
            'most_common_event': max(event_counts.items(), key=lambda x: x[1]) if event_counts else None
        }

# 告警处理器
class AlertHandler:
    """告警处理器基类"""
    
    def handle_alert(self, alert: Dict[str, Any]):
        """处理告警"""
        raise NotImplementedError

class EmailAlertHandler(AlertHandler):
    """邮件告警处理器"""
    
    def __init__(self, smtp_config: Dict[str, str], recipients: List[str]):
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    def handle_alert(self, alert: Dict[str, Any]):
        """发送邮件告警"""
        subject = f"Gold-Seeker安全告警: {alert['alert_type']}"
        body = self._format_alert_email(alert)
        
        # 发送邮件逻辑
        self._send_email(subject, body, self.recipients)
    
    def _format_alert_email(self, alert: Dict[str, Any]) -> str:
        """格式化告警邮件"""
        return f"""
        安全告警通知
        
        告警类型: {alert['alert_type']}
        严重程度: {alert['severity']}
        时间: {alert['timestamp']}
        详情: {json.dumps(alert['details'], indent=2, ensure_ascii=False)}
        
        请及时处理此安全事件。
        """
    
    def _send_email(self, subject: str, body: str, recipients: List[str]):
        """发送邮件"""
        # 实现邮件发送逻辑
        pass

class SlackAlertHandler(AlertHandler):
    """Slack告警处理器"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def handle_alert(self, alert: Dict[str, Any]):
        """发送Slack告警"""
        message = self._format_slack_message(alert)
        self._send_slack_message(message)
    
    def _format_slack_message(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """格式化Slack消息"""
        return {
            "text": f"🚨 Gold-Seeker安全告警: {alert['alert_type']}",
            "attachments": [
                {
                    "color": self._get_color_by_severity(alert['severity']),
                    "fields": [
                        {
                            "title": "严重程度",
                            "value": alert['severity'],
                            "short": True
                        },
                        {
                            "title": "时间",
                            "value": alert['timestamp'],
                            "short": True
                        },
                        {
                            "title": "详情",
                            "value": json.dumps(alert['details'], indent=2, ensure_ascii=False),
                            "short": False
                        }
                    ]
                }
            ]
        }
    
    def _get_color_by_severity(self, severity: str) -> str:
        """根据严重程度获取颜色"""
        color_map = {
            'HIGH': 'danger',
            'MEDIUM': 'warning',
            'LOW': 'good'
        }
        return color_map.get(severity, 'good')
    
    def _send_slack_message(self, message: Dict[str, Any]):
        """发送Slack消息"""
        # 实现Slack消息发送逻辑
        pass
```

## 📋 合规要求

### GDPR合规

```python
# GDPR合规服务
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class GDPRComplianceService:
    """GDPR合规服务"""
    
    def __init__(self):
        self.consent_records = {}
        self.data_processing_records = {}
        self.data_subject_requests = []
    
    def record_consent(self, user_id: str, consent_data: Dict[str, Any]):
        """记录用户同意"""
        consent_record = {
            'user_id': user_id,
            'consent_given': True,
            'timestamp': datetime.utcnow().isoformat(),
            'consent_data': consent_data,
            'ip_address': get_client_ip(),
            'user_agent': get_user_agent()
        }
        
        self.consent_records[user_id] = consent_record
    
    def withdraw_consent(self, user_id: str):
        """撤回同意"""
        if user_id in self.consent_records:
            self.consent_records[user_id]['consent_given'] = False
            self.consent_records[user_id]['withdrawal_timestamp'] = datetime.utcnow().isoformat()
    
    def has_consent(self, user_id: str) -> bool:
        """检查是否有有效同意"""
        if user_id not in self.consent_records:
            return False
        
        return self.consent_records[user_id]['consent_given']
    
    def record_data_processing(self, 
                             user_id: str, 
                             processing_type: str, 
                             purpose: str, 
                             legal_basis: str):
        """记录数据处理活动"""
        record = {
            'user_id': user_id,
            'processing_type': processing_type,
            'purpose': purpose,
            'legal_basis': legal_basis,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if user_id not in self.data_processing_records:
            self.data_processing_records[user_id] = []
        
        self.data_processing_records[user_id].append(record)
    
    def handle_data_subject_request(self, 
                                   request_type: str, 
                                   user_id: str, 
                                   request_data: Dict[str, Any]):
        """处理数据主体请求"""
        request = {
            'request_id': self._generate_request_id(),
            'request_type': request_type,  # ACCESS, CORRECTION, DELETION, PORTABILITY
            'user_id': user_id,
            'request_data': request_data,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'PENDING'
        }
        
        self.data_subject_requests.append(request)
        
        # 处理请求
        if request_type == 'ACCESS':
            self._handle_access_request(request)
        elif request_type == 'CORRECTION':
            self._handle_correction_request(request)
        elif request_type == 'DELETION':
            self._handle_deletion_request(request)
        elif request_type == 'PORTABILITY':
            self._handle_portability_request(request)
        
        return request['request_id']
    
    def _handle_access_request(self, request: Dict[str, Any]):
        """处理访问请求"""
        user_id = request['user_id']
        
        # 收集用户的所有个人数据
        personal_data = self._collect_personal_data(user_id)
        
        # 准备响应
        response = {
            'request_id': request['request_id'],
            'personal_data': personal_data,
            'processing_activities': self.data_processing_records.get(user_id, []),
            'consent_records': self.consent_records.get(user_id, {}),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # 发送响应给用户
        self._send_data_access_response(user_id, response)
        
        # 更新请求状态
        request['status'] = 'COMPLETED'
        request['completion_timestamp'] = datetime.utcnow().isoformat()
    
    def _handle_deletion_request(self, request: Dict[str, Any]):
        """处理删除请求"""
        user_id = request['user_id']
        
        # 删除用户数据
        self._delete_user_data(user_id)
        
        # 更新请求状态
        request['status'] = 'COMPLETED'
        request['completion_timestamp'] = datetime.utcnow().isoformat()
    
    def _delete_user_data(self, user_id: str):
        """删除用户数据"""
        # 删除同意记录
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        # 删除数据处理记录
        if user_id in self.data_processing_records:
            del self.data_processing_records[user_id]
        
        # 删除其他个人数据
        # 这里需要根据实际的数据存储结构来实现
        pass
    
    def _collect_personal_data(self, user_id: str) -> Dict[str, Any]:
        """收集用户的个人数据"""
        # 这里需要根据实际的数据存储结构来实现
        return {}
    
    def _generate_request_id(self) -> str:
        """生成请求ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _send_data_access_response(self, user_id: str, response: Dict[str, Any]):
        """发送数据访问响应"""
        # 实现响应发送逻辑
        pass
```

## 🛡️ 安全最佳实践

### 开发安全

```python
# 安全编码实践
import re
import html
from typing import Any, Dict, List

class SecurityValidator:
    """安全验证器"""
    
    @staticmethod
    def sanitize_input(input_data: str) -> str:
        """清理输入数据"""
        # HTML转义
        sanitized = html.escape(input_data)
        
        # 移除潜在的危险字符
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """验证密码强度"""
        result = {
            'is_valid': True,
            'errors': []
        }
        
        if len(password) < 8:
            result['is_valid'] = False
            result['errors'].append('密码长度至少8位')
        
        if not re.search(r'[A-Z]', password):
            result['is_valid'] = False
            result['errors'].append('密码必须包含大写字母')
        
        if not re.search(r'[a-z]', password):
            result['is_valid'] = False
            result['errors'].append('密码必须包含小写字母')
        
        if not re.search(r'\d', password):
            result['is_valid'] = False
            result['errors'].append('密码必须包含数字')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result['is_valid'] = False
            result['errors'].append('密码必须包含特殊字符')
        
        return result
    
    @staticmethod
    def validate_file_upload(file_data: bytes, filename: str) -> Dict[str, Any]:
        """验证文件上传"""
        result = {
            'is_valid': True,
            'errors': []
        }
        
        # 检查文件大小
        if len(file_data) > 10 * 1024 * 1024:  # 10MB
            result['is_valid'] = False
            result['errors'].append('文件大小超过限制')
        
        # 检查文件扩展名
        allowed_extensions = ['.csv', '.xlsx', '.json', '.txt']
        file_extension = '.' + filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            result['is_valid'] = False
            result['errors'].append('不支持的文件类型')
        
        # 检查文件内容
        if b'<script' in file_data.lower():
            result['is_valid'] = False
            result['errors'].append('文件包含潜在恶意内容')
        
        return result

# 安全配置
class SecurityConfig:
    """安全配置"""
    
    # 密码策略
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    
    # 会话配置
    SESSION_TIMEOUT = 3600  # 1小时
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 900  # 15分钟
    
    # 文件上传配置
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = ['.csv', '.xlsx', '.json', '.txt']
    
    # API配置
    API_RATE_LIMIT = 100  # 每分钟请求数
    API_TIMEOUT = 30  # 秒
    
    # 加密配置
    ENCRYPTION_ALGORITHM = 'AES-256-GCM'
    HASH_ALGORITHM = 'SHA-256'
    KEY_DERIVATION_ITERATIONS = 100000
```

### 运维安全

```bash
#!/bin/bash
# 安全运维脚本

# 1. 系统更新
update_system() {
    echo "更新系统..."
    apt update && apt upgrade -y
    apt autoremove -y
    apt autoclean
}

# 2. 防火墙配置
configure_firewall() {
    echo "配置防火墙..."
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw enable
}

# 3. SSL证书配置
configure_ssl() {
    echo "配置SSL证书..."
    certbot --nginx -d your-domain.com
    certbot renew --dry-run
}

# 4. 日志轮转
configure_log_rotation() {
    echo "配置日志轮转..."
    cat > /etc/logrotate.d/gold-seeker << EOF
/var/log/gold-seeker/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 gold-seeker gold-seeker
    postrotate
        systemctl reload gold-seeker
    endscript
}
EOF
}

# 5. 备份配置
configure_backup() {
    echo "配置备份..."
    cat > /etc/cron.daily/gold-seeker-backup << EOF
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/gold-seeker"
mkdir -p \$BACKUP_DIR

# 备份数据库
pg_dump gold_seeker > \$BACKUP_DIR/db_\$DATE.sql

# 备份数据文件
tar -czf \$BACKUP_DIR/data_\$DATE.tar.gz /var/lib/gold-seeker/data

# 上传到云存储
aws s3 cp \$BACKUP_DIR/db_\$DATE.sql s3://gold-seeker-backups/database/
aws s3 cp \$BACKUP_DIR/data_\$DATE.tar.gz s3://gold-seeker-backups/data/

# 清理旧备份
find \$BACKUP_DIR -name "*.sql" -mtime +30 -delete
find \$BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
EOF

    chmod +x /etc/cron.daily/gold-seeker-backup
}

# 6. 安全扫描
security_scan() {
    echo "执行安全扫描..."
    
    # 检查开放端口
    nmap -sS -O localhost
    
    # 检查漏洞
    lynis audit system
    
    # 检查文件权限
    find /var/lib/gold-seeker -type f -perm /o+w -ls
}

# 执行所有安全配置
main() {
    update_system
    configure_firewall
    configure_ssl
    configure_log_rotation
    configure_backup
    security_scan
    
    echo "安全配置完成!"
}

main "$@"
```

---

通过实施这些安全措施和最佳实践，Gold-Seeker平台可以建立全面的安全防护体系，确保数据和系统的安全性、完整性和可用性。安全是一个持续的过程，需要定期评估和改进。