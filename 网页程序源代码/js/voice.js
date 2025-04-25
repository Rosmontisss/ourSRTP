// voice.js - 语音处理相关功能

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化语音采集界面
    initVoiceCollection();
    
    // 设置录音按钮事件
    setupRecordingButtons();
    
    // 设置帮助按钮事件
    setupHelpButton();
    
    // 设置历史录音项事件
    setupRecordingHistory();
});

// 初始化语音采集界面
function initVoiceCollection() {
    // 检查浏览器是否支持Web Audio API
    if (!window.AudioContext) {
        alert('您的浏览器不支持语音录制功能，请升级到最新版本的浏览器。');
        return;
    }
    
    // 初始化录音状态
    const statusElement = document.getElementById('recording-status');
    if (statusElement) {
        statusElement.textContent = '准备就绪';
    }
}

// 设置录音按钮事件
function setupRecordingButtons() {
    const startBtn = document.getElementById('start-recording');
    const reRecordBtn = document.getElementById('re-record');
    const uploadBtn = document.getElementById('upload-recording');
    const cancelBtn = document.getElementById('cancel-recording');
    const progressBar = document.getElementById('recording-progress');
    const statusElement = document.getElementById('recording-status');
    
    let isRecording = false;
    let audioContext;
    let mediaRecorder;
    let audioChunks = [];
    
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            if (isRecording) return;
            
            // 请求麦克风权限
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    audioContext = new AudioContext();
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = function(e) {
                        audioChunks.push(e.data);
                    };
                    
                    mediaRecorder.onstop = function() {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        // 创建音频播放器
                        const audio = new Audio(audioUrl);
                        // 可以在这里添加音频播放功能
                        
                        // 更新UI状态
                        statusElement.textContent = '录音完成';
                        reRecordBtn.disabled = false;
                        uploadBtn.disabled = false;
                        cancelBtn.disabled = true;
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    
                    // 更新UI状态
                    statusElement.textContent = '正在录音...';
                    startBtn.disabled = true;
                    reRecordBtn.disabled = false;
                    uploadBtn.disabled = true;
                    cancelBtn.disabled = false;
                    
                    // 更新进度条
                    updateRecordingProgress(progressBar, mediaRecorder);
                })
                .catch(function(err) {
                    console.error('无法访问麦克风:', err);
                    statusElement.textContent = '麦克风访问失败';
                });
        });
    }
    
    if (reRecordBtn) {
        reRecordBtn.addEventListener('click', function() {
            if (!isRecording) {
                // 重置录音状态
                audioChunks = [];
                statusElement.textContent = '准备就绪';
                reRecordBtn.disabled = true;
                uploadBtn.disabled = true;
                cancelBtn.disabled = true;
                startBtn.disabled = false;
                progressBar.style.width = '0%';
            }
        });
    }
    
    if (uploadBtn) {
        uploadBtn.addEventListener('click', function() {
            if (audioChunks.length > 0) {
                // 创建音频Blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // 模拟上传到服务器
                uploadAudioToServer(audioBlob)
                    .then(function(response) {
                        alert('语音上传成功，分析结果即将显示！');
                        window.location.href = 'voice-analysis.html';
                    })
                    .catch(function(error) {
                        console.error('上传失败:', error);
                        alert('语音上传失败，请重试');
                    });
            }
        });
    }
    
    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            if (mediaRecorder) {
                mediaRecorder.stop();
                isRecording = false;
            }
        });
    }
    
    // 更新录音进度条
    function updateRecordingProgress(progressElement, recorder) {
        let startTime = Date.now();
        
        const updateInterval = setInterval(function() {
            if (!isRecording) {
                clearInterval(updateInterval);
                return;
            }
            
            const elapsedTime = Date.now() - startTime;
            const progressPercentage = Math.min(100, (elapsedTime / 180000) * 100); // 3分钟限制
            progressElement.style.width = progressPercentage + '%';
            
            // 更新时间显示
            const timeDisplay = document.querySelector('.time-display');
            if (timeDisplay) {
                const minutes = Math.floor(elapsedTime / 60000);
                const seconds = Math.floor((elapsedTime % 60000) / 1000);
                timeDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }
    
    // 模拟上传音频到服务器
    function uploadAudioToServer(blob) {
        return new Promise(function(resolve, reject) {
            // 实际应用中，这里应该发送到真实服务器
            setTimeout(function() {
                resolve({ success: true, message: '上传成功' });
            }, 1500);
        });
    }
}

// 设置帮助按钮事件
function setupHelpButton() {
    const helpBtn = document.getElementById('help-btn');
    const helpContent = document.getElementById('help-content');
    
    if (helpBtn && helpContent) {
        helpBtn.addEventListener('click', function() {
            helpContent.classList.toggle('visible');
        });
        
        const closeBtn = document.getElementById('close-help');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                helpContent.classList.remove('visible');
            });
        }
    }
}

// 设置历史录音项事件
function setupRecordingHistory() {
    const playButtons = document.querySelectorAll('.play-btn');
    const deleteButtons = document.querySelectorAll('.delete-btn');
    const viewButtons = document.querySelectorAll('.view-btn');
    
    playButtons.forEach(function(btn) {
        btn.addEventListener('click', function() {
            // 播放录音
            console.log('播放录音');
        });
    });
    
    deleteButtons.forEach(function(btn) {
        btn.addEventListener('click', function() {
            // 删除录音
            if (confirm('确定要删除这条录音吗？')) {
                console.log('删除录音');
                // 移除UI元素
                this.closest('.history-item').remove();
            }
        });
    });
    
    viewButtons.forEach(function(btn) {
        btn.addEventListener('click', function() {
            // 查看分析结果
            window.location.href = 'voice-analysis.html';
        });
    });
}