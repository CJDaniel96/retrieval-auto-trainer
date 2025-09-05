/**
 * 國際化 (i18n) 多語言支援
 * Internationalization support for multiple languages
 */

const translations = {
    'zh-TW': {
        // Header
        'app.title': 'AI 自動化訓練系統',
        'header.total_tasks': '總任務數',
        'header.active_tasks': '執行中',
        'header.success_rate': '成功率',
        
        // Navigation Tabs
        'tab.new_training': '新增訓練',
        'tab.task_list': '任務列表',
        'tab.settings': '系統設定',
        
        // New Training Form
        'form.start_new_training': '啟動新的訓練任務',
        'form.input_dir': '輸入資料夾路徑',
        'form.input_dir_placeholder': '/path/to/input/folder',
        'form.output_dir': '輸出資料夾路徑',
        'form.output_dir_placeholder': '/path/to/output/folder',
        'form.site': '地區名稱',
        'form.line_id': '產線ID',
        'form.advanced_settings': '進階設定',
        'form.max_epochs': '最大訓練輪數',
        'form.batch_size': '批次大小',
        'form.learning_rate': '學習率',
        'form.start_training': '開始訓練',
        'form.submitting': '提交中...',
        
        // Task Status
        'status.pending': '等待中',
        'status.pending_orientation': '等待方向確認',
        'status.running': '執行中',
        'status.completed': '已完成',
        'status.failed': '失敗',
        
        // Task Actions
        'action.view_detail': '查看詳情',
        'action.confirm_orientation': '確認方向',
        'action.download': '下載',
        'action.cancel': '取消',
        'action.refresh': '刷新列表',
        
        // Settings
        'settings.api_server': 'API伺服器位址',
        'settings.current_config': '當前配置',
        'settings.reload_config': '重新載入配置',
        
        // Messages
        'msg.training_started': '訓練任務已啟動！',
        'msg.task_cancelled': '任務已取消',
        'msg.no_tasks': '暫無訓練任務',
        'msg.loading': '載入中...',
        'msg.api_connection_failed': '無法連線到API伺服器',
        'msg.start_time': '開始時間',
        'msg.new_task': '新增任務',
        
        // Orientation Confirmation
        'orientation.title': '影像方向確認',
        'orientation.description': '請為每個類別選擇正確的影像方向，這將用於後續的資料增強',
        'orientation.task_id': '任務ID',
        'orientation.return_home': '返回主頁',
        'orientation.samples': '個樣本',
        'orientation.up': '上方',
        'orientation.down': '下方',
        'orientation.left': '左方',
        'orientation.right': '右方',
        'orientation.confirmed': '已確認',
        'orientation.total': '總共',
        'orientation.classes': '個類別',
        'orientation.confirm_all': '確認所有方向並開始訓練',
        'orientation.submitting': '提交中...',
        'orientation.success': '方向確認成功！正在開始訓練...',
        'orientation.loading_samples': '載入樣本影像中...',
        'orientation.load_failed': '載入樣本失敗',
        'orientation.retry': '重試'
    },
    
    'zh-CN': {
        // Header
        'app.title': 'AI 自动化训练系统',
        'header.total_tasks': '总任务数',
        'header.active_tasks': '执行中',
        'header.success_rate': '成功率',
        
        // Navigation Tabs
        'tab.new_training': '新增训练',
        'tab.task_list': '任务列表',
        'tab.settings': '系统设置',
        
        // New Training Form
        'form.start_new_training': '启动新的训练任务',
        'form.input_dir': '输入文件夹路径',
        'form.input_dir_placeholder': '/path/to/input/folder',
        'form.output_dir': '输出文件夹路径',
        'form.output_dir_placeholder': '/path/to/output/folder',
        'form.site': '地区名称',
        'form.line_id': '产线ID',
        'form.advanced_settings': '高级设置',
        'form.max_epochs': '最大训练轮数',
        'form.batch_size': '批次大小',
        'form.learning_rate': '学习率',
        'form.start_training': '开始训练',
        'form.submitting': '提交中...',
        
        // Task Status
        'status.pending': '等待中',
        'status.pending_orientation': '等待方向确认',
        'status.running': '执行中',
        'status.completed': '已完成',
        'status.failed': '失败',
        
        // Task Actions
        'action.view_detail': '查看详情',
        'action.confirm_orientation': '确认方向',
        'action.download': '下载',
        'action.cancel': '取消',
        'action.refresh': '刷新列表',
        
        // Settings
        'settings.api_server': 'API服务器地址',
        'settings.current_config': '当前配置',
        'settings.reload_config': '重新加载配置',
        
        // Messages
        'msg.training_started': '训练任务已启动！',
        'msg.task_cancelled': '任务已取消',
        'msg.no_tasks': '暂无训练任务',
        'msg.loading': '加载中...',
        'msg.api_connection_failed': '无法连接到API服务器',
        'msg.start_time': '开始时间',
        'msg.new_task': '新增任务',
        
        // Orientation Confirmation
        'orientation.title': '图像方向确认',
        'orientation.description': '请为每个类别选择正确的图像方向，这将用于后续的数据增强',
        'orientation.task_id': '任务ID',
        'orientation.return_home': '返回主页',
        'orientation.samples': '个样本',
        'orientation.up': '上方',
        'orientation.down': '下方',
        'orientation.left': '左方',
        'orientation.right': '右方',
        'orientation.confirmed': '已确认',
        'orientation.total': '总共',
        'orientation.classes': '个类别',
        'orientation.confirm_all': '确认所有方向并开始训练',
        'orientation.submitting': '提交中...',
        'orientation.success': '方向确认成功！正在开始训练...',
        'orientation.loading_samples': '加载样本图像中...',
        'orientation.load_failed': '加载样本失败',
        'orientation.retry': '重试'
    },
    
    'en': {
        // Header
        'app.title': 'AI Automated Training System',
        'header.total_tasks': 'Total Tasks',
        'header.active_tasks': 'Active',
        'header.success_rate': 'Success Rate',
        
        // Navigation Tabs
        'tab.new_training': 'New Training',
        'tab.task_list': 'Task List',
        'tab.settings': 'Settings',
        
        // New Training Form
        'form.start_new_training': 'Start New Training Task',
        'form.input_dir': 'Input Directory Path',
        'form.input_dir_placeholder': '/path/to/input/folder',
        'form.output_dir': 'Output Directory Path',
        'form.output_dir_placeholder': '/path/to/output/folder',
        'form.site': 'Site Name',
        'form.line_id': 'Line ID',
        'form.advanced_settings': 'Advanced Settings',
        'form.max_epochs': 'Max Epochs',
        'form.batch_size': 'Batch Size',
        'form.learning_rate': 'Learning Rate',
        'form.start_training': 'Start Training',
        'form.submitting': 'Submitting...',
        
        // Task Status
        'status.pending': 'Pending',
        'status.pending_orientation': 'Awaiting Orientation',
        'status.running': 'Running',
        'status.completed': 'Completed',
        'status.failed': 'Failed',
        
        // Task Actions
        'action.view_detail': 'View Details',
        'action.confirm_orientation': 'Confirm Orientation',
        'action.download': 'Download',
        'action.cancel': 'Cancel',
        'action.refresh': 'Refresh List',
        
        // Settings
        'settings.api_server': 'API Server Address',
        'settings.current_config': 'Current Configuration',
        'settings.reload_config': 'Reload Configuration',
        
        // Messages
        'msg.training_started': 'Training task started!',
        'msg.task_cancelled': 'Task cancelled',
        'msg.no_tasks': 'No training tasks',
        'msg.loading': 'Loading...',
        'msg.api_connection_failed': 'Unable to connect to API server',
        'msg.start_time': 'Start Time',
        'msg.new_task': 'New Task',
        
        // Orientation Confirmation
        'orientation.title': 'Image Orientation Confirmation',
        'orientation.description': 'Please select the correct image orientation for each class, which will be used for subsequent data augmentation',
        'orientation.task_id': 'Task ID',
        'orientation.return_home': 'Return Home',
        'orientation.samples': 'samples',
        'orientation.up': 'Up',
        'orientation.down': 'Down',
        'orientation.left': 'Left',
        'orientation.right': 'Right',
        'orientation.confirmed': 'Confirmed',
        'orientation.total': 'Total',
        'orientation.classes': 'classes',
        'orientation.confirm_all': 'Confirm All Orientations and Start Training',
        'orientation.submitting': 'Submitting...',
        'orientation.success': 'Orientation confirmation successful! Starting training...',
        'orientation.loading_samples': 'Loading sample images...',
        'orientation.load_failed': 'Failed to load samples',
        'orientation.retry': 'Retry'
    },
    
    'vi': {
        // Header
        'app.title': 'Hệ Thống Huấn Luyện AI Tự Động',
        'header.total_tasks': 'Tổng Nhiệm Vụ',
        'header.active_tasks': 'Đang Chạy',
        'header.success_rate': 'Tỷ Lệ Thành Công',
        
        // Navigation Tabs
        'tab.new_training': 'Huấn Luyện Mới',
        'tab.task_list': 'Danh Sách Nhiệm Vụ',
        'tab.settings': 'Cài Đặt',
        
        // New Training Form
        'form.start_new_training': 'Bắt Đầu Nhiệm Vụ Huấn Luyện Mới',
        'form.input_dir': 'Đường Dẫn Thư Mục Đầu Vào',
        'form.input_dir_placeholder': '/path/to/input/folder',
        'form.output_dir': 'Đường Dẫn Thư Mục Đầu Ra',
        'form.output_dir_placeholder': '/path/to/output/folder',
        'form.site': 'Tên Địa Điểm',
        'form.line_id': 'ID Dây Chuyền',
        'form.advanced_settings': 'Cài Đặt Nâng Cao',
        'form.max_epochs': 'Số Epoch Tối Đa',
        'form.batch_size': 'Kích Thước Batch',
        'form.learning_rate': 'Tốc Độ Học',
        'form.start_training': 'Bắt Đầu Huấn Luyện',
        'form.submitting': 'Đang gửi...',
        
        // Task Status
        'status.pending': 'Đang Chờ',
        'status.pending_orientation': 'Chờ Xác Nhận Hướng',
        'status.running': 'Đang Chạy',
        'status.completed': 'Hoàn Thành',
        'status.failed': 'Thất Bại',
        
        // Task Actions
        'action.view_detail': 'Xem Chi Tiết',
        'action.confirm_orientation': 'Xác Nhận Hướng',
        'action.download': 'Tải Xuống',
        'action.cancel': 'Hủy',
        'action.refresh': 'Làm Mới Danh Sách',
        
        // Settings
        'settings.api_server': 'Địa Chỉ Máy Chủ API',
        'settings.current_config': 'Cấu Hình Hiện Tại',
        'settings.reload_config': 'Tải Lại Cấu Hình',
        
        // Messages
        'msg.training_started': 'Nhiệm vụ huấn luyện đã bắt đầu!',
        'msg.task_cancelled': 'Nhiệm vụ đã bị hủy',
        'msg.no_tasks': 'Không có nhiệm vụ huấn luyện',
        'msg.loading': 'Đang tải...',
        'msg.api_connection_failed': 'Không thể kết nối đến máy chủ API',
        'msg.start_time': 'Thời Gian Bắt Đầu',
        'msg.new_task': 'Nhiệm Vụ Mới',
        
        // Orientation Confirmation
        'orientation.title': 'Xác Nhận Hướng Hình Ảnh',
        'orientation.description': 'Vui lòng chọn hướng hình ảnh đúng cho mỗi lớp, điều này sẽ được sử dụng cho việc tăng cường dữ liệu sau này',
        'orientation.task_id': 'ID Nhiệm Vụ',
        'orientation.return_home': 'Về Trang Chủ',
        'orientation.samples': 'mẫu',
        'orientation.up': 'Lên',
        'orientation.down': 'Xuống',
        'orientation.left': 'Trái',
        'orientation.right': 'Phải',
        'orientation.confirmed': 'Đã xác nhận',
        'orientation.total': 'Tổng cộng',
        'orientation.classes': 'lớp',
        'orientation.confirm_all': 'Xác Nhận Tất Cả Hướng và Bắt Đầu Huấn Luyện',
        'orientation.submitting': 'Đang gửi...',
        'orientation.success': 'Xác nhận hướng thành công! Đang bắt đầu huấn luyện...',
        'orientation.loading_samples': 'Đang tải hình ảnh mẫu...',
        'orientation.load_failed': 'Tải mẫu thất bại',
        'orientation.retry': 'Thử Lại'
    }
};

// i18n Manager Class
class I18nManager {
    constructor() {
        this.currentLanguage = this.getStoredLanguage() || this.detectLanguage();
        this.translations = translations;
    }

    detectLanguage() {
        const browserLang = navigator.language || navigator.userLanguage;
        
        // Map browser languages to our supported languages
        if (browserLang.startsWith('zh-TW') || browserLang.startsWith('zh-Hant')) {
            return 'zh-TW';
        } else if (browserLang.startsWith('zh-CN') || browserLang.startsWith('zh-Hans') || browserLang.startsWith('zh')) {
            return 'zh-CN';
        } else if (browserLang.startsWith('vi')) {
            return 'vi';
        } else {
            return 'en'; // Default to English
        }
    }

    getStoredLanguage() {
        return localStorage.getItem('app_language');
    }

    setLanguage(language) {
        if (this.translations[language]) {
            this.currentLanguage = language;
            localStorage.setItem('app_language', language);
            this.updatePageContent();
            // Trigger language change event
            window.dispatchEvent(new CustomEvent('languageChanged', { detail: language }));
        }
    }

    t(key, fallback = null) {
        const translation = this.translations[this.currentLanguage]?.[key] || 
                          this.translations['en']?.[key] || 
                          fallback || 
                          key;
        return translation;
    }

    updatePageContent() {
        // Update all elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            // Update different types of content
            if (element.tagName === 'INPUT' && (element.type === 'text' || element.type === 'email')) {
                element.placeholder = translation;
            } else if (element.tagName === 'INPUT' && element.type === 'submit') {
                element.value = translation;
            } else {
                element.textContent = translation;
            }
        });

        // Update title
        const titleElement = document.querySelector('title');
        if (titleElement) {
            titleElement.textContent = this.t('app.title');
        }

        // Update page title in header
        const headerTitle = document.querySelector('.logo span');
        if (headerTitle) {
            headerTitle.textContent = this.t('app.title');
        }
    }

    getAvailableLanguages() {
        return {
            'zh-TW': '繁體中文',
            'zh-CN': '简体中文', 
            'en': 'English',
            'vi': 'Tiếng Việt'
        };
    }

    getCurrentLanguage() {
        return this.currentLanguage;
    }
}

// Create global i18n instance
window.i18n = new I18nManager();

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.i18n.updatePageContent();
});