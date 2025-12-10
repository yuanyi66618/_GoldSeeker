# Gold-Seeker Makefile
# 提供常用的开发和部署命令

.PHONY: help install install-dev test test-cov lint format clean build docs run-example

# 默认目标
help:
	@echo "Gold-Seeker 开发工具"
	@echo ""
	@echo "可用命令:"
	@echo "  install      - 安装生产依赖"
	@echo "  install-dev  - 安装开发依赖"
	@echo "  test         - 运行测试"
	@echo "  test-cov     - 运行测试并生成覆盖率报告"
	@echo "  lint         - 代码检查"
	@echo "  format       - 代码格式化"
	@echo "  clean        - 清理临时文件"
	@echo "  build        - 构建包"
	@echo "  docs         - 生成文档"
	@echo "  run-example  - 运行示例"
	@echo "  all          - 执行完整的CI流程"

# 安装依赖
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# 测试
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=agents --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

test-integration:
	pytest tests/ -v -m integration

# 代码质量
lint:
	flake8 agents/ tests/ examples/
	mypy agents/
	black --check agents/ tests/ examples/
	isort --check-only agents/ tests/ examples/

format:
	black agents/ tests/ examples/
	isort agents/ tests/ examples/

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf logs/
	rm -rf output/
	rm -rf .cache/

# 构建
build: clean
	python setup.py sdist bdist_wheel

# 文档
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# 示例
run-example:
	python examples/complete_workflow.py

run-synthetic:
	python examples/complete_workflow.py --synthetic

# 开发环境
dev-setup: install-dev
	cp .env.example .env
	@echo "请编辑 .env 文件，添加你的API密钥"

# 数据准备
data-download:
	@echo "下载示例数据..."
	mkdir -p data/examples
	# 这里可以添加数据下载命令

# 性能测试
benchmark:
	python -m pytest tests/test_performance.py -v

# 安全检查
security:
	bandit -r agents/
	safety check

# 依赖检查
deps-check:
	pip-audit

# 版本管理
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

# 发布
release: clean test lint build
	@echo "准备发布新版本..."
	@echo "请确保已更新版本号和CHANGELOG"

# Docker相关
docker-build:
	docker build -t gold-seeker .

docker-run:
	docker run -it --rm gold-seeker

# CI流程
all: clean lint test-cov docs
	@echo "CI流程完成"

# 开发服务器
dev-server:
	python -m agents.cli --debug

# 监控
monitor:
	python -m agents.cli monitor

# 备份
backup:
	tar -czf gold-seeker-backup-$(shell date +%Y%m%d).tar.gz \
		--exclude=.git \
		--exclude=__pycache__ \
		--exclude=.pytest_cache \
		--exclude=logs \
		--exclude=output \
		--exclude=.cache \
		.

# 恢复
restore:
	@echo "恢复备份: $(BACKUP_FILE)"
	tar -xzf $(BACKUP_FILE)

# 性能分析
profile:
	python -m cProfile -o profile.stats examples/complete_workflow.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# 内存分析
memory-profile:
	mprof run examples/complete_workflow.py
	mprof plot

# 并行测试
test-parallel:
	pytest tests/ -v -n auto

# 持续集成模拟
ci: clean lint test-cov security deps-check
	@echo "CI检查通过"

# 部署准备
deploy-prepare: clean test lint build
	@echo "部署准备完成"

# 本地开发
dev: install-dev
	@echo "开发环境设置完成"
	@echo "运行 'make run-example' 测试示例"

# 生产环境
prod: install
	@echo "生产环境设置完成"