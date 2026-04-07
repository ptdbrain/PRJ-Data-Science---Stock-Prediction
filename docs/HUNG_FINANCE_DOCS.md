# Bao cao ky thuat: Pipeline du lieu bao cao tai chinh

Tai lieu nay mo ta co che hoat dong, cau truc du lieu va cac thong so xu ly cho pipeline bao cao tai chinh trong nhanh Hung.

## 1) Muc tieu nghiep vu

Khoi xu ly bao cao tai chinh duoc thiet ke de:

- Thu thap du lieu BCTC theo quy cho ma co phieu TCB.
- Chuan hoa du lieu ve dang tinh toan theo quy.
- Tao cac bien tai chinh phuc vu mo hinh hoa du bao.
- Luu bo du lieu da xu ly vao clean_finance cho cac buoc feature merge va model training.

## 2) Thanh phan he thong

Pipeline gom 2 module van hanh:

- data_collection/collect_finance.py
- preprocessing/process_finance.py

Quan he giua 2 module:

1. collect_finance lay va ghi du lieu goc vao raw_finance.
2. process_finance doc raw_finance, bien doi, tinh toan va ghi clean_finance.

## 3) Nguon du lieu va tham so thu thap

Du lieu dau vao duoc lay qua thu vien vnstock, voi cau hinh tu he thong:

- ma co phieu: TCB
- nguon du lieu: VCI
- tan suat bao cao: quy
- cua so lay lieu: 12 quy

Ba nhom bao cao duoc truy van:

- Income Statement
- Balance Sheet
- Cash Flow Statement

Sau khi lay xong, du lieu duoc noi theo chieu dong bang pd.concat va ghi vao raw_finance.

## 4) Mo hinh du lieu

### 4.1 Dinh dang du lieu goc

Du lieu o raw_finance duoc ky vong theo long format, toi thieu gom:

- ticker
- quarter
- metric_name
- value

Long format giup de them metric moi, nhung chua thuan tien cho tinh ratio truc tiep.

### 4.2 Dinh dang sau bien doi

Buoc preprocessing chuyen raw_finance sang wide format voi nguyen tac:

- index: ticker, quarter
- columns: metric_name
- values: value

Sau pivot, moi dong bieu dien mot ma trong mot quy, co cac cot metric rieng de tinh chi so.

## 5) Quy trinh xu ly chi tiet

### Buoc A - Tai du lieu

- Doc bang raw_finance.
- Ghi log so dong dau vao.

### Buoc B - Chuan hoa theo quy

- Pivot long -> wide de tao mat tran chi so theo quy.
- Sap xep theo ticker, quarter de dam bao thu tu thoi gian.

### Buoc C - Tinh bien tai chinh

Nhom chi so cot loi:

- ROE = Net Income / Equity
- ROA = Net Income / Total Assets

Nhom chi so dac thu ngan hang:

- NIM = Net Interest Income / Total Assets
- NPL = Bad Debt / Total Loans

Neu thieu cot thanh phan trong cong thuc, gia tri duoc gan NA thay vi ep tinh.

### Buoc D - Tinh toc do tang truong theo nam

Voi moi cot so, he thong tao them cot tang truong nam:

- ten cot moi: <metric>_YoY_Growth
- cong thuc: pct_change(periods=4)

Ly do chon periods=4: du lieu theo quy, 4 quy tuong ung 1 nam.

### Buoc E - Xu ly gia tri thieu

- Ap dung forward fill theo tung ticker.
- Gioi han bu filling: toi da 1 quy lien tiep.

Co che nay giam mat du lieu ngan han nhung han che lan truyen sai so khi khoang trong qua dai.

### Buoc F - Luu ket qua

- Ghi bang clean_finance trong SQLite.
- Xuat tep clean_finance.csv.
- Neu CSV dang bi khoa, ghi sang clean_finance_temp.csv.

## 6) Danh muc bien dau ra

Nhom bien thuong xuyen xuat hien o clean_finance:

- Khoa chinh nghiep vu: ticker, quarter
- Metric goc sau pivot: tuy vao du lieu tra ve tu vnstock
- Chi so tinh toan truc tiep: ROE, ROA, NIM, NPL
- Chi so tang truong: cac cot hau to _YoY_Growth cho bien so

## 7) Kiem soat chat luong du lieu

Nhung diem can kiem tra khi van hanh:

- Ton tai du 4 cot long-format truoc pivot.
- Quy tac dat ten quarter thong nhat de tranh sap xep sai trinh tu.
- So luong cot metric sau pivot phai phu hop ky vong theo tung ky lay lieu.
- Gia tri chia so khong bang 0 khi tinh ratio (neu bang 0 can xu ly null an toan).
- Ti le NA truoc va sau fill de danh gia do day du cua bo du lieu.

## 8) Gioi han hien tai cua pipeline

Tai thoi diem lap bao cao, bo ratio hien hanh tap trung vao nhom can ban va ngan hang. Cac chi so nghiep vu thuong gap khac (vi du PE, PB, Debt to Equity, Cost to Income) can duoc bo sung cong thuc va mapping cot dau vao neu muon su dung trong model theo danh muc day du.

## 9) Huong dan tai hien

Trinh tu chay:

1. python -m database.schema
2. python -m data_collection.collect_finance
3. python -m preprocessing.process_finance

Kiem tra dau ra:

- raw_finance co du lieu goc theo quy
- clean_finance co cot ratio va cot YoY growth
- Co tep clean_finance.csv hoac clean_finance_temp.csv

## 10) Tom tat ky thuat

Pipeline bao cao tai chinh duoc xay dung theo huong tách ro dau vao, bien doi va dau ra. Du lieu duoc lay theo quy tu nguon ben ngoai, chuan hoa thanh bang theo quy, tinh cac bien tai chinh then chot va bo sung bien tang truong nam. Dau ra clean_finance duoc thiet ke de dung truc tiep cho cac buoc tong hop dac trung va huan luyen mo hinh du bao.
