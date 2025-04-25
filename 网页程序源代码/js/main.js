// main.js - 主脚本

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化用户信息
    initUserInfo();
    
    // 设置退出按钮事件
    setupLogoutButton();
    
    // 设置音频播放器功能
    setupAudioPlayer();
    
    // 设置日程表功能
    setupSchedule();
    
    // 设置待办事项功能
    setupTodoList();
    
    // 设置通知功能
    setupNotifications();
});

// 初始化用户信息
function initUserInfo() {
    // 从本地存储或API获取用户信息
    const userInfo = {
        name: '张三',
        avatar: 'assets/images/default-avatar.png',
        userType: 'patient' // 'patient' 或 'doctor'
    };
    
    // 更新用户信息显示
    document.getElementById('user-name').textContent = userInfo.name;
    const avatarElement = document.getElementById('user-avatar');
    if (avatarElement && avatarElement.querySelector('img')) {
        avatarElement.querySelector('img').src = userInfo.avatar;
    }
    
    // 根据用户类型设置颜色主题
    if (userInfo.userType === 'doctor') {
        document.documentElement.style.setProperty('--patient-primary', '#0984e3');
        document.documentElement.style.setProperty('--patient-secondary', '#74b9ff');
        document.documentElement.style.setProperty('--patient-accent', '#00cec9');
        document.documentElement.style.setProperty('--patient-warm-bg', '#f0f7ff');
        document.documentElement.style.setProperty('--patient-text', '#2d3436');
    }
}

// 设置退出按钮事件
function setupLogoutButton() {
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            // 清除用户信息
            localStorage.removeItem('userInfo');
            
            // 跳转到登录页面
            window.location.href = 'login.html';
        });
    }
}

// 设置音频播放器功能
function setupAudioPlayer() {
    const playPauseBtn = document.querySelector('.play-pause');
    const progressBar = document.querySelector('.audio-player .progress');
    const prevBtn = document.querySelector('.control-btn.prev');
    const nextBtn = document.querySelector('.control-btn.next');
    
    let isPlaying = false;
    
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', function() {
            isPlaying = !isPlaying;
            const icon = playPauseBtn.querySelector('.icon');
            
            if (isPlaying) {
                icon.className = 'icon pause';
                // 模拟进度条动画
                animateProgressBar(progressBar);
            } else {
                icon.className = 'icon play';
                // 停止进度条动画
                if (progressBar.style.animation) {
                    progressBar.style.animation = 'none';
                }
            }
        });
    }
    
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            // 切换到上一首
            console.log('切换到上一首');
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            // 切换到下一首
            console.log('切换到下一首');
        });
    }
    
    // 进度条动画
    function animateProgressBar(element) {
        element.style.animation = 'progress 30s linear infinite';
    }
}

// 设置日程表功能
function setupSchedule() {
    const prevBtn = document.querySelector('.date-nav.prev');
    const nextBtn = document.querySelector('.date-nav.next');
    const currentDate = document.querySelector('.current-date');
    const addAppointmentBtn = document.querySelector('.add-appointment');
    
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            // 上个月
            console.log('切换到上个月');
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            // 下个月
            console.log('切换到下个月');
        });
    }
    
    if (addAppointmentBtn) {
        addAppointmentBtn.addEventListener('click', function() {
            // 添加新预约
            alert('添加新预约功能将在此实现');
        });
    }
}

// 设置待办事项功能
function setupTodoList() {
    const addTodoBtn = document.querySelector('.add-todo');
    const todoItems = document.querySelectorAll('.todo-item input[type="checkbox"]');
    
    if (addTodoBtn) {
        addTodoBtn.addEventListener('click', function() {
            // 添加新待办
            alert('添加新待办功能将在此实现');
        });
    }
    
    todoItems.forEach(function(checkbox) {
        checkbox.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (this.checked) {
                label.style.textDecoration = 'line-through';
                label.style.color = '#adb5bd';
            } else {
                label.style.textDecoration = 'none';
                label.style.color = '';
            }
        });
    });
}

// 设置通知功能
function setupNotifications() {
    const notificationItems = document.querySelectorAll('.notification-item');
    
    notificationItems.forEach(function(item) {
        item.addEventListener('click', function() {
            // 标记为已读
            this.classList.add('read');
        });
    });
}