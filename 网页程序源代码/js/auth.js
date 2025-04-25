// auth.js - 认证相关功能

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 设置登录表单提交事件
    setupLoginForm();
    
    // 设置登录选项卡切换
    setupLoginTabs();
    
    // 设置欢迎文案随机显示
    setupRandomWelcomeText();
});

// 设置登录表单提交事件
function setupLoginForm() {
    const loginForm = document.getElementById('login-form');
    const loginTabs = document.querySelectorAll('.login-tab');
    
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const userType = document.querySelector('.login-tab.active').dataset.userType;
            
            // 验证表单
            if (!username || !password) {
                alert('请输入用户名和密码');
                return;
            }
            
            // 模拟登录请求
            simulateLogin(username, password, userType);
        });
    }
}

// 模拟登录请求
function simulateLogin(username, password, userType) {
    // 在实际应用中，这里应该发送到后端进行验证
    console.log('登录请求:', { username, password, userType });
    
    // 模拟网络延迟
    setTimeout(function() {
        // 模拟登录成功
        const userInfo = {
            username: username,
            userType: userType
        };
        
        // 存储用户信息
        localStorage.setItem('userInfo', JSON.stringify(userInfo));
        
        // 跳转到主界面
        window.location.href = 'dashboard.html';
    }, 1500);
}

// 设置登录选项卡切换
function setupLoginTabs() {
    const tabs = document.querySelectorAll('.login-tab');
    
    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            // 移除所有选项卡的active类
            tabs.forEach(function(t) {
                t.classList.remove('active');
            });
            
            // 为当前选项卡添加active类
            this.classList.add('active');
        });
    });
}

// 设置欢迎文案随机显示
function setupRandomWelcomeText() {
    const welcomeTexts = [
        '欢迎回到抑郁症语音筛查平台（SRTP测试版），您的心理健康管家',
        '抑郁症语音筛查平台（SRTP测试版），让您的每一天都充满阳光',
        '倾听您的声音，守护您的心灵',
        '心理健康，从这里开始',
        '平静心情，拥抱美好'
    ];
    
    const welcomeTextElement = document.getElementById('welcome-text');
    
    if (welcomeTextElement) {
        // 随机选择一条欢迎文案
        const randomIndex = Math.floor(Math.random() * welcomeTexts.length);
        welcomeTextElement.textContent = welcomeTexts[randomIndex];
        
        // 设置自动更换文案
        setInterval(function() {
            const newIndex = Math.floor(Math.random() * welcomeTexts.length);
            welcomeTextElement.textContent = welcomeTexts[newIndex];
        }, 5000);
    }
}

// 检查用户登录状态
function checkLoginStatus() {
    const userInfo = localStorage.getItem('userInfo');
    return userInfo ? JSON.parse(userInfo) : null;
}

// 重定向未登录用户
function redirectIfNotLoggedIn() {
    const userInfo = checkLoginStatus();
    if (!userInfo) {
        window.location.href = 'login.html';
    }
}