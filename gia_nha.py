import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
np.random.seed(42) 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print("--- Đang tải và tiền xử lý dữ liệu giá nhà ---")

duong_dan_tai_du_lieu = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

bang_du_lieu = pd.read_csv(duong_dan_tai_du_lieu)

bang_du_lieu.dropna(inplace=True)

cot_vi_tri_ma_hoa = pd.get_dummies(bang_du_lieu['ocean_proximity'], prefix='vi_tri')
bang_du_lieu = pd.concat([bang_du_lieu.drop('ocean_proximity', axis=1), cot_vi_tri_ma_hoa], axis=1)

cac_cot_dac_trung = bang_du_lieu.drop(columns=['median_house_value']).columns
X_dac_trung = bang_du_lieu[cac_cot_dac_trung].values.astype(np.float32)

y_muc_tieu = (bang_du_lieu['median_house_value'].values.reshape(-1, 1).astype(np.float32)) / 100000.0

X_huan_luyen, X_kiem_tra, y_huan_luyen, y_kiem_tra = train_test_split(
    X_dac_trung, y_muc_tieu, test_size=0.2, random_state=42, shuffle=True
)
print("Shape của X_dac_trung:", X_dac_trung.shape)
bo_chuan_hoa = StandardScaler()
X_huan_luyen = bo_chuan_hoa.fit_transform(X_huan_luyen)
X_kiem_tra = bo_chuan_hoa.transform(X_kiem_tra)


class MangNoRonHoiQuy:
    def __init__(self, so_dau_vao, so_noron_an=32, toc_do_hoc=0.001, quan_tinh=0.9):
        self.so_dau_vao = so_dau_vao
        self.so_noron_an = so_noron_an
        self.toc_do_hoc = toc_do_hoc 
        self.quan_tinh = quan_tinh   

        self.W1 = np.random.randn(so_dau_vao, so_noron_an) * np.sqrt(2. / so_dau_vao)
        self.b1 = np.zeros((1, so_noron_an))
        self.W2 = np.random.randn(so_noron_an, 1) * np.sqrt(2. / so_noron_an)
        self.b2 = np.zeros((1, 1))

        self.van_toc_W1 = np.zeros_like(self.W1)
        self.van_toc_b1 = np.zeros_like(self.b1)
        self.van_toc_W2 = np.zeros_like(self.W2)
        self.van_toc_b2 = np.zeros_like(self.b2)

    def ham_kich_hoat_relu(self, z):
        return np.maximum(0, z)

    def dao_ham_relu(self, z):
        return (z > 0).astype(float)

    def lan_truyen_xuoi(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.ham_kich_hoat_relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2   
        return self.Z2

    def lan_truyen_nguoc(self, X, y_thuc_te, y_du_doan):
        tong_so_mau = X.shape[0]

        dZ2 = (2 / tong_so_mau) * (y_du_doan - y_thuc_te)
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.dao_ham_relu(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Cập nhật lại Trọng số (W) và Độ lệch (b)
        self.van_toc_W1 = self.quan_tinh * self.van_toc_W1 - self.toc_do_hoc * dW1
        self.van_toc_b1 = self.quan_tinh * self.van_toc_b1 - self.toc_do_hoc * db1
        self.W1 += self.van_toc_W1
        self.b1 += self.van_toc_b1

        self.van_toc_W2 = self.quan_tinh * self.van_toc_W2 - self.toc_do_hoc * dW2
        self.van_toc_b2 = self.quan_tinh * self.van_toc_b2 - self.toc_do_hoc * db2
        self.W2 += self.van_toc_W2
        self.b2 += self.van_toc_b2

    def huan_luyen(self, X, y, so_vong_lap=1000, hien_thi_log=True):
        lich_su_sai_so = []
        for vong in range(so_vong_lap + 1):
            y_du_doan = self.lan_truyen_xuoi(X)
            sai_so = np.mean((y_du_doan - y) ** 2) 
            lich_su_sai_so.append(sai_so)
            
            self.lan_truyen_nguoc(X, y, y_du_doan)

            if hien_thi_log and vong % 100 == 0:
                print(f"Vòng lặp {vong:4d} | Sai số MSE: {sai_so:.4f}")
        return lich_su_sai_so

    def du_doan(self, X):
        return self.lan_truyen_xuoi(X)
        
    def luu_mo_hinh(self, duong_dan):
        np.savez(duong_dan, 
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 v_W1=self.van_toc_W1, v_b1=self.van_toc_b1, 
                 v_W2=self.van_toc_W2, v_b2=self.van_toc_b2)
        print(f" Đã lưu mô hình mạng nơ-ron tại: '{duong_dan}'")

    def tai_mo_hinh(self, duong_dan):
        du_lieu = np.load(duong_dan)
        self.W1 = du_lieu['W1']
        self.b1 = du_lieu['b1']
        self.W2 = du_lieu['W2']
        self.b2 = du_lieu['b2']
        self.van_toc_W1 = du_lieu['v_W1']
        self.van_toc_b1 = du_lieu['v_b1']
        self.van_toc_W2 = du_lieu['v_W2']
        self.van_toc_b2 = du_lieu['v_b2']
        print(f" Đã nạp lại thành công toàn bộ cấu trúc mạng nơ-ron từ: '{duong_dan}'")

# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH
def danh_gia_mo_hinh(ten_mo_hinh, y_thuc_te, y_du_doan):
    mse = mean_squared_error(y_thuc_te, y_du_doan)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_thuc_te, y_du_doan)
    r2 = r2_score(y_thuc_te, y_du_doan)
    print(f"{ten_mo_hinh:25s} -> MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
print("\n--- Bắt đầu huấn luyện các mô hình ---")

# 3.1. Hồi quy tuyến tính
mo_hinh_tuyen_tinh = LinearRegression()
mo_hinh_tuyen_tinh.fit(X_huan_luyen, y_huan_luyen)
y_du_doan_tuyen_tinh = mo_hinh_tuyen_tinh.predict(X_kiem_tra)
danh_gia_mo_hinh("Hồi quy tuyến tính", y_kiem_tra, y_du_doan_tuyen_tinh)

# 3.2. Rừng ngẫu nhiên
mo_hinh_rung_ngau_nhien = RandomForestRegressor(n_estimators=100, random_state=42)
mo_hinh_rung_ngau_nhien.fit(X_huan_luyen, y_huan_luyen.ravel())
y_du_doan_rung_ngau_nhien = mo_hinh_rung_ngau_nhien.predict(X_kiem_tra)
danh_gia_mo_hinh("Rừng ngẫu nhiên", y_kiem_tra, y_du_doan_rung_ngau_nhien)

# 3.3. Mạng Nơ-ron Đa tầng (MLP)
print("\n[Đang huấn luyện Mạng Nơ-ron...  ]")
mo_hinh_mang_noron = MangNoRonHoiQuy(so_dau_vao=X_huan_luyen.shape[1], so_noron_an=64, toc_do_hoc=0.01)
lich_su_sai_so = mo_hinh_mang_noron.huan_luyen(X_huan_luyen, y_huan_luyen, so_vong_lap=1000, hien_thi_log=False) 
y_du_doan_noron = mo_hinh_mang_noron.du_doan(X_kiem_tra)
danh_gia_mo_hinh("Mạng Nơ-ron", y_kiem_tra, y_du_doan_noron)


print("\n--- Đang tiến hành lưu mô hình ---")
os.makedirs("models", exist_ok=True)
mo_hinh_mang_noron.luu_mo_hinh("models/mang_noron_gia_nha.npz")
joblib.dump(bo_chuan_hoa, "models/scaler.pkl")
print(" Đã lưu bộ chuẩn hóa dữ liệu tại: 'models/scaler.pkl'")


# Nạp lại mô hình đã lưu
print("\n---  Nạp lại mô hình đã lưu  ---")

mo_hinh_dung_ngay = MangNoRonHoiQuy(so_dau_vao=X_huan_luyen.shape[1], so_noron_an=64, toc_do_hoc=0.01)
mo_hinh_dung_ngay.tai_mo_hinh("models/mang_noron_gia_nha.npz") 

y_du_doan_dung_ngay = mo_hinh_dung_ngay.du_doan(X_kiem_tra)
danh_gia_mo_hinh("Mạng Nơ-ron ", y_kiem_tra, y_du_doan_dung_ngay)


#   Nạp lại mô hình đã lưu để HUẤN LUYỆN TIẾP 
print("\n--- đang nạp lại mô hình đã lưu để HUẤN LUYỆN TIẾP ---")

mo_hinh_train_tiep = MangNoRonHoiQuy(so_dau_vao=X_huan_luyen.shape[1], so_noron_an=64, toc_do_hoc=0.01)
mo_hinh_train_tiep.tai_mo_hinh("models/mang_noron_gia_nha.npz")

print("[Đang huấn luyện tiếp thêm 200 vòng lặp...]")
mo_hinh_train_tiep.huan_luyen(X_huan_luyen, y_huan_luyen, so_vong_lap=200, hien_thi_log=True)

y_du_doan_train_tiep = mo_hinh_train_tiep.du_doan(X_kiem_tra)
danh_gia_mo_hinh("Mạng Nơ-ron (Sau huấn luyện tiếp)", y_kiem_tra, y_du_doan_train_tiep)
mo_hinh_train_tiep.luu_mo_hinh("models/mang_noron_gia_nha_nang_cap.npz")


#  ỨNG DỤNG THỰC TẾ: ĐỊNH GIÁ MỘT CĂN NHÀ BẤT KỲ 

print("\n===  ỨNG DỤNG THỰC TẾ: ĐỊNH GIÁ CĂN NHÀ MỚI ===")

thong_tin_nha_moi = np.array([[ 
    -122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 
    0, 0, 0, 1, 0 
]], dtype=np.float32)

bo_scaler_tai_lai = joblib.load("models/scaler.pkl")
nha_moi_chuan_hoa = bo_scaler_tai_lai.transform(thong_tin_nha_moi)

# Gọi mô hình 'mo_hinh_dung_ngay' 
gia_raw = mo_hinh_dung_ngay.du_doan(nha_moi_chuan_hoa)[0][0]
gia_tien_thuc_te = gia_raw * 100000.0

print(f"\n[KẾT QUẢ ĐỊNH GIÁ TỪ HỆ THỐNG]:")
print(f" Căn nhà có thông số trên được dự đoán có giá là: {gia_tien_thuc_te:,.0f} USD")


# Vẽ biểu đồ Sai số
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(lich_su_sai_so, color='green', linewidth=2)
plt.title("Biểu đồ giảm sai số - Dự đoán Giá Nhà")
plt.xlabel("Số vòng lặp (Epoch)")
plt.ylabel("Sai số MSE")
plt.grid(True)
plt.savefig("results/loss_curve_gianha.png", dpi=150)
print("\nĐã lưu biểu đồ Sai số tại thư mục 'results/loss_curve_gianha.png'")