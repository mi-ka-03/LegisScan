class LegalDocChecker {
    constructor() {
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.uploadBox = document.getElementById('uploadBox');
        this.fileInput = document.getElementById('fileInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.status = document.getElementById('status');
        this.error = document.getElementById('error');
        this.results = document.getElementById('results');
        this.errorCount = document.getElementById('errorCount');
        this.errorList = document.getElementById('errorList');
    }

    bindEvents() {
        // 点击上传区域触发文件选择
        this.uploadBox.addEventListener('click', () => {
            this.fileInput.click();
        });

        // 拖拽上传
        this.uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadBox.style.borderColor = '#2980b9';
            this.uploadBox.style.background = '#f8f9fa';
        });

        this.uploadBox.addEventListener('dragleave', () => {
            this.uploadBox.style.borderColor = '#3498db';
            this.uploadBox.style.background = 'white';
        });

        this.uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadBox.style.borderColor = '#3498db';
            this.uploadBox.style.background = 'white';
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // 文件选择变化
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // 分析按钮点击
        this.analyzeBtn.addEventListener('click', () => {
            this.analyzeDocument();
        });
    }

    handleFileSelect(file) {
        // 验证文件类型
        const validTypes = ['.txt', '.pdf', '.docx'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        
        if (!validTypes.includes(fileExtension)) {
            this.showError('请上传 .txt, .pdf 或 .docx 格式的文件');
            return;
        }

        this.currentFile = file;
        this.analyzeBtn.disabled = false;
        this.hideError();
        this.hideResults();

        // 更新上传框显示
        this.uploadBox.innerHTML = `
            <div style="text-align: center;">
                <div style="font-size: 3em; margin-bottom: 10px;">📄</div>
                <p><strong>${file.name}</strong></p>
                <p class="support-text">${this.formatFileSize(file.size)}</p>
            </div>
        `;
    }

    async analyzeDocument() {
        if (!this.currentFile) return;

        this.showLoading();
        this.hideError();
        this.hideResults();

        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            // 这里替换成你后端的实际地址
            const response = await fetch('http://localhost:8000/check', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`服务器错误: ${response.status}`);
            }

            const results = await response.json();
            this.displayResults(results);

        } catch (err) {
            this.showError('分析失败: ' + err.message);
            console.error('分析错误:', err);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(results) {
        if (!results || !results.errors) {
            this.showError('未收到有效分析结果');
            return;
        }

        this.errorCount.textContent = results.errors.length;

        // 清空之前的错误列表
        this.errorList.innerHTML = '';

        // 添加新的错误项
        results.errors.forEach((error, index) => {
            const errorItem = document.createElement('div');
            errorItem.className = 'error-item';
            errorItem.innerHTML = `
                <div class="error-type">${error.type || '未知错误'}</div>
                <div class="error-position">位置: ${error.position || '未知'}</div>
                <div class="error-message">${error.message || '无详细描述'}</div>
            `;
            this.errorList.appendChild(errorItem);
        });

        this.results.hidden = false;
    }

    showLoading() {
        this.status.hidden = false;
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.status.hidden = true;
    }

    showError(message) {
        this.error.textContent = message;
        this.error.hidden = false;
    }

    hideError() {
        this.error.hidden = true;
    }

    hideResults() {
        this.results.hidden = true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new LegalDocChecker();
});