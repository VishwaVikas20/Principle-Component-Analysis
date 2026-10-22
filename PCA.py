import numpy as np
class PCA:
  def __init__(self,l):
    self.D=None
    self.l=l
    self.mean=None
    self.error=0
  def fit(self,X):
    #X is (m,n) matrix with n features m samples
    Xc=X-X.mean(axis=0)
    #using .reshape(-1,1) because without it is is (n,) order matrix rather than (n,1)
    self.mean=X.mean(axis=0).reshape(-1,1)
    eigenvalues,eigenvectors=np.linalg.eigh(Xc.T@Xc)
    idex=np.argsort(eigenvalues)[::-1][:self.l]
    self.D=eigenvectors[:,idex]
    self.error=(np.linalg.norm(Xc-Xc@self.D@self.D.T))**2
  def f(self,x):
    #x is column vector
    return self.D.T@(x-self.mean)
  def g(self,c):
    # c is a column vector
    return self.D@c+self.mean
    
X = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]])
model = PCA(1)
model.fit(X)
print(model.error)

x = np.array([[1.0], [11.0]])
c = model.f(x)
print(c.shape)
print(c)
print(model.g(c))
