#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
#include "ms/linalg/linalg.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <functional>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace ml {

// ========================== Utilities ==========================

Mat mat_zeros(int r, int c) { return Mat(r, Vec(c, 0.0)); }
Mat mat_eye(int n) {
    auto M = mat_zeros(n, n);
    for (int i=0;i<n;++i) M[i][i]=1.0;
    return M;
}
Mat mat_T(const Mat& A) {
    if (A.empty()) return {};
    int m=A.size(), n=A[0].size();
    Mat B(n, Vec(m));
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) B[j][i]=A[i][j];
    return B;
}
Mat mat_mul(const Mat& A, const Mat& B) {
    int m=A.size(), k=A[0].size(), n=B[0].size();
    Mat C(m, Vec(n,0));
    for (int i=0;i<m;++i) for (int l=0;l<k;++l) for (int j=0;j<n;++j)
        C[i][j]+=A[i][l]*B[l][j];
    return C;
}
Vec mat_vec(const Mat& A, const Vec& x) {
    int m=A.size(), n=x.size();
    Vec y(m,0);
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) y[i]+=A[i][j]*x[j];
    return y;
}
Vec vec_add(const Vec& a, const Vec& b) {
    Vec c(a.size()); for (size_t i=0;i<a.size();++i) c[i]=a[i]+b[i]; return c;
}
Vec vec_sub(const Vec& a, const Vec& b) {
    Vec c(a.size()); for (size_t i=0;i<a.size();++i) c[i]=a[i]-b[i]; return c;
}
Vec vec_scale(double s, const Vec& v) {
    Vec c(v.size()); for (size_t i=0;i<v.size();++i) c[i]=s*v[i]; return c;
}
double vec_dot(const Vec& a, const Vec& b) {
    double s=0; for (size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s;
}
double vec_norm(const Vec& v) { return std::sqrt(vec_dot(v,v)); }

static double sq_eucl_dist(const Vec& a, const Vec& b) {
    double s=0; for (size_t i=0;i<a.size();++i) { double d=a[i]-b[i]; s+=d*d; }
    return s;
}
static double eucl_dist(const Vec& a, const Vec& b) {
    return std::sqrt(sq_eucl_dist(a, b));
}

// Add bias column to design matrix
static Mat add_bias(const Mat& X) {
    Mat Xb(X.size(), Vec(X[0].size()+1));
    for (size_t i=0;i<X.size();++i) {
        Xb[i][0]=1.0;
        for (size_t j=0;j<X[i].size();++j) Xb[i][j+1]=X[i][j];
    }
    return Xb;
}

// Solve Ax=b via Gauss-Jordan
static Vec gauss_solve(Mat A, Vec b) {
    int n=A.size();
    for (int col=0;col<n;++col) {
        int p=col;
        for (int r=col+1;r<n;++r) if (std::abs(A[r][col])>std::abs(A[p][col])) p=r;
        std::swap(A[col],A[p]); std::swap(b[col],b[p]);
        double sc=A[col][col];
        if (std::abs(sc)<1e-14) continue;
        for (int j=col;j<n;++j) A[col][j]/=sc; b[col]/=sc;
        for (int r=0;r<n;++r) if (r!=col) {
            double f=A[r][col];
            for (int j=col;j<n;++j) A[r][j]-=f*A[col][j];
            b[r]-=f*b[col];
        }
    }
    return b;
}

// ========================== Linear Regression ==========================

void LinearRegression::fit(const Mat& X, const Vec& y, bool use_intercept) {
    Mat Xb = use_intercept ? add_bias(X) : X;
    auto Xt = mat_T(Xb);
    auto XtX = mat_mul(Xt, Xb);
    auto Xty = mat_vec(Xt, y);
    auto params = gauss_solve(XtX, Xty);
    if (use_intercept) {
        intercept = params[0];
        coef.assign(params.begin()+1, params.end());
    } else { coef = params; intercept = 0; }
}
Vec LinearRegression::predict(const Mat& X) const {
    Vec p(X.size());
    for (size_t i=0;i<X.size();++i) { p[i]=intercept+vec_dot(coef,X[i]); }
    return p;
}
double LinearRegression::score(const Mat& X, const Vec& y) const { return r2_score(predict(X),y); }

// ========================== Ridge Regression ==========================

void RidgeRegression::fit(const Mat& X, const Vec& y, bool use_intercept) {
    Mat Xb = use_intercept ? add_bias(X) : X;
    int n=Xb[0].size();
    auto Xt = mat_T(Xb);
    auto XtX = mat_mul(Xt, Xb);
    for (int i=1;i<n;++i) XtX[i][i]+=alpha;  // don't regularise intercept
    auto Xty = mat_vec(Xt, y);
    auto params = gauss_solve(XtX, Xty);
    if (use_intercept) {
        intercept=params[0]; coef.assign(params.begin()+1,params.end());
    } else { coef=params; intercept=0; }
}
Vec RidgeRegression::predict(const Mat& X) const {
    Vec p(X.size()); for (size_t i=0;i<X.size();++i) p[i]=intercept+vec_dot(coef,X[i]); return p;
}
double RidgeRegression::score(const Mat& X, const Vec& y) const { return r2_score(predict(X),y); }

// ========================== Lasso (coordinate descent) ==========================

static double soft_threshold(double x, double t) {
    if (x > t) return x-t;
    if (x < -t) return x+t;
    return 0.0;
}

void LassoRegression::fit(const Mat& X, const Vec& y) {
    int n=X.size(), p=X[0].size();
    coef.assign(p, 0.0); intercept=0;
    Vec r=y;
    for (int iter=0;iter<max_iter;++iter) {
        double max_change=0;
        // intercept
        double mu=0; for (double v:r) mu+=v; mu/=n;
        intercept+=mu; for (double& rv:r) rv-=mu;
        for (int j=0;j<p;++j) {
            // Add back j's contribution
            for (int i=0;i<n;++i) r[i]+=coef[j]*X[i][j];
            // Compute rho
            double rho=0, Xjn=0;
            for (int i=0;i<n;++i) { rho+=X[i][j]*r[i]; Xjn+=X[i][j]*X[i][j]; }
            double new_coef = soft_threshold(rho/n, alpha) / (Xjn/n + 1e-12);
            double diff=std::abs(new_coef-coef[j]);
            max_change=std::max(max_change,diff);
            for (int i=0;i<n;++i) r[i]-=new_coef*X[i][j];
            coef[j]=new_coef;
        }
        if (max_change<1e-6) break;
    }
}
Vec LassoRegression::predict(const Mat& X) const {
    Vec p(X.size()); for (size_t i=0;i<X.size();++i) p[i]=intercept+vec_dot(coef,X[i]); return p;
}

// ========================== Elastic Net (coordinate descent) ==========================

void ElasticNet::fit(const Mat& X, const Vec& y) {
    coef.clear(); intercept=0.0;
    if (X.empty() || y.empty() || X.size()!=y.size() || X[0].empty()) return;
    int n=X.size(), p=X[0].size();
    coef.assign(p, 0.0); intercept=0;
    double l1=alpha*l1_ratio, l2=alpha*(1.0-l1_ratio);
    Vec r=y;
    for (int iter=0;iter<max_iter;++iter) {
        double max_change=0;
        // intercept
        double mu=0; for (double v:r) mu+=v; mu/=n;
        intercept+=mu; for (double& rv:r) rv-=mu;
        for (int j=0;j<p;++j) {
            // Add back j's contribution
            for (int i=0;i<n;++i) r[i]+=coef[j]*X[i][j];
            // Compute rho
            double rho=0, Xjn=0;
            for (int i=0;i<n;++i) { rho+=X[i][j]*r[i]; Xjn+=X[i][j]*X[i][j]; }
            double new_coef = soft_threshold(rho/n, l1) / (Xjn/n + l2 + 1e-12);
            double diff=std::abs(new_coef-coef[j]);
            max_change=std::max(max_change,diff);
            for (int i=0;i<n;++i) r[i]-=new_coef*X[i][j];
            coef[j]=new_coef;
        }
        if (max_change<1e-6) break;
    }
}
Vec ElasticNet::predict(const Mat& X) const {
    Vec p(X.size()); for (size_t i=0;i<X.size();++i) p[i]=intercept+vec_dot(coef,X[i]); return p;
}

// ========================== Logistic Regression (GD) ==========================

static double sigmoid(double z) { return 1.0/(1.0+std::exp(-z)); }

void LogisticRegression::fit(const Mat& X, const Vec& y) {
    int n=X.size(), p=X[0].size();
    coef.assign(p,0.0); intercept=0;
    double lr=0.1/C;
    for (int iter=0;iter<max_iter;++iter) {
        Vec grad_coef(p,0); double grad_int=0;
        for (int i=0;i<n;++i) {
            double z=intercept+vec_dot(coef,X[i]);
            double err=sigmoid(z)-y[i];
            grad_int+=err;
            for (int j=0;j<p;++j) grad_coef[j]+=err*X[i][j];
        }
        intercept-=lr*grad_int/n;
        for (int j=0;j<p;++j) coef[j]-=lr*(grad_coef[j]/n + coef[j]/C);
    }
}
Vec LogisticRegression::predict_proba(const Mat& X) const {
    Vec p(X.size());
    for (size_t i=0;i<X.size();++i) p[i]=sigmoid(intercept+vec_dot(coef,X[i]));
    return p;
}
Vec LogisticRegression::predict(const Mat& X) const {
    auto p=predict_proba(X);
    for (auto& v:p) v=v>=0.5?1.0:0.0;
    return p;
}
double LogisticRegression::score(const Mat& X, const Vec& y) const { return accuracy(predict(X),y); }

// ========================== KNN ==========================

void KNN::fit(const Mat& X, const Vec& y) { X_train=X; y_train=y; }
Vec KNN::predict(const Mat& X) const {
    Vec pred(X.size());
    for (size_t i=0;i<X.size();++i) {
        std::vector<std::pair<double,double>> dists;
        for (size_t j=0;j<X_train.size();++j)
            dists.push_back({eucl_dist(X[i],X_train[j]), y_train[j]});
        std::sort(dists.begin(),dists.end());
        // Majority vote
        std::map<double,int> counts;
        for (int l=0;l<k && l<(int)dists.size();++l) counts[dists[l].second]++;
        auto it=std::max_element(counts.begin(),counts.end(),[](auto&a,auto&b){return a.second<b.second;});
        pred[i]=it->first;
    }
    return pred;
}
double KNN::score(const Mat& X, const Vec& y) const { return accuracy(predict(X),y); }

// ========================== Naive Bayes (Gaussian) ==========================

void NaiveBayes::fit(const Mat& X, const Vec& y) {
    std::set<double> cls(y.begin(),y.end());
    classes.assign(cls.begin(),cls.end());
    int C=classes.size(), n=X.size(), p=X[0].size();
    mean.assign(C,Vec(p,0)); var.assign(C,Vec(p,0)); class_prior.assign(C,0);
    std::vector<int> counts(C,0);
    std::vector<int> class_idx(n);
    for (int i=0;i<n;++i) {
        int ci=static_cast<int>(std::find(classes.begin(),classes.end(),y[i])-classes.begin());
        class_idx[i]=ci;
        counts[ci]++;
        for (int j=0;j<p;++j) mean[ci][j]+=X[i][j];
    }
    for (int c=0;c<C;++c) { class_prior[c]=(double)counts[c]/n; for (int j=0;j<p;++j) mean[c][j]/=counts[c]; }
    for (int i=0;i<n;++i) {
        const int ci=class_idx[i];
        for (int j=0;j<p;++j) {
            const double diff=X[i][j]-mean[ci][j];
            var[ci][j]+=diff*diff;
        }
    }
    for (int c=0;c<C;++c) for (int j=0;j<p;++j) var[c][j]=var[c][j]/counts[c]+1e-9;
}
Vec NaiveBayes::predict(const Mat& X) const {
    Vec pred(X.size());
    if (X.empty()) return pred;
    const int C=static_cast<int>(classes.size()), p=static_cast<int>(X[0].size());
    Vec log_prior(C);
    Mat inv_var(C, Vec(p)), log_norm(C, Vec(p));
    for (int c=0;c<C;++c) {
        log_prior[c]=std::log(class_prior[c]);
        for (int j=0;j<p;++j) {
            inv_var[c][j]=1.0/var[c][j];
            log_norm[c][j]=0.5*std::log(2*M_PI*var[c][j]);
        }
    }
    for (size_t i=0;i<X.size();++i) {
        double best=-1e300; int best_c=0;
        for (int c=0;c<C;++c) {
            double log_prob=log_prior[c];
            for (int j=0;j<p;++j) {
                const double diff=X[i][j]-mean[c][j];
                log_prob-=log_norm[c][j]+0.5*diff*diff*inv_var[c][j];
            }
            if (log_prob>best){best=log_prob;best_c=c;}
        }
        pred[i]=classes[best_c];
    }
    return pred;
}
double NaiveBayes::score(const Mat& X, const Vec& y) const { return accuracy(predict(X),y); }

// ========================== LDA / QDA helpers ==========================

static Mat mat_add_reg(const Mat& A, double reg) {
    Mat B=A;
    for (size_t i=0;i<B.size();++i) B[i][i]+=reg;
    return B;
}

static Mat mat_inv(const Mat& A, double reg=0.0) {
    int n=(int)A.size();
    if (n==0) return {};
    Mat inv(n, Vec(n,0.0));
    Mat base=mat_add_reg(A, reg);
    for (int j=0;j<n;++j) {
        Mat Acopy=base;
        Vec e(n,0.0); e[j]=1.0;
        Vec col=gauss_solve(Acopy, e);
        for (int i=0;i<n;++i) inv[i][j]=col[i];
    }
    return inv;
}

static double mat_quad_form(const Mat& A, const Vec& x) {
    Vec Ax=mat_vec(A, x);
    return vec_dot(x, Ax);
}

static double mat_logdet_spd(Mat A, double reg=0.0) {
    int n=(int)A.size();
    if (n==0) return 0.0;
    for (int i=0;i<n;++i) A[i][i]+=reg;
    Mat L(n, Vec(n,0.0));
    for (int i=0;i<n;++i) {
        for (int j=0;j<=i;++j) {
            double s=A[i][j];
            for (int k=0;k<j;++k) s-=L[i][k]*L[j][k];
            if (i==j) {
                if (s<=1e-14) return -1e300;
                L[i][j]=std::sqrt(s);
            } else {
                L[i][j]=s/L[j][j];
            }
        }
    }
    double logdet=0.0;
    for (int i=0;i<n;++i) logdet+=2.0*std::log(L[i][i]);
    return logdet;
}

static void sym_eig_power(const Mat& A, Mat& evecs, Vec& evals, int k, int max_iter=200) {
    int n=(int)A.size();
    k=std::min(k, n);
    evecs.assign(k, Vec(n,0.0));
    evals.assign(k, 0.0);
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0,1);
    for (int r=0;r<k;++r) {
        Vec v(n);
        for (auto& x:v) x=nd(rng);
        double nn=vec_norm(v);
        if (nn<1e-14) { v[0]=1.0; nn=1.0; }
        for (auto& x:v) x/=nn;
        for (int it=0;it<max_iter;++it) {
            Vec u(n,0.0);
            for (int i=0;i<n;++i)
                for (int j=0;j<n;++j)
                    u[i]+=A[i][j]*v[j];
            for (int prev=0;prev<r;++prev) {
                double d=vec_dot(evecs[prev], u);
                for (int i=0;i<n;++i) u[i]-=d*evecs[prev][i];
            }
            nn=vec_norm(u);
            if (nn<1e-14) break;
            for (auto& x:u) x/=nn;
            v=u;
        }
        Vec Au(n,0.0);
        for (int i=0;i<n;++i)
            for (int j=0;j<n;++j)
                Au[i]+=A[i][j]*v[j];
        double lambda=vec_dot(v, Au);
        evals[r]=lambda;
        evecs[r]=v;
    }
}

static Mat class_scatter(const Mat& X, const Vec& mean) {
    int p=(int)mean.size();
    Mat S=mat_zeros(p, p);
    for (const auto& x:X) {
        Vec d=vec_sub(x, mean);
        for (int i=0;i<p;++i)
            for (int j=0;j<p;++j)
                S[i][j]+=d[i]*d[j];
    }
    return S;
}

// ========================== Linear Discriminant Analysis ==========================

void LDA::fit(const Mat& X, const Vec& y) {
    classes.clear(); class_prior.clear(); mean.clear();
    pooled_cov_inv.clear(); discrim_const.clear(); discrim_coef.clear();
    projection.clear(); overall_mean.clear();
    if (X.empty()) return;

    std::set<double> cls(y.begin(), y.end());
    classes.assign(cls.begin(), cls.end());
    int C=(int)classes.size(), n=(int)X.size(), p=(int)X[0].size();
    if (C<2) return;

    mean.assign(C, Vec(p,0.0));
    class_prior.assign(C, 0.0);
    std::vector<int> counts(C, 0);

    overall_mean.assign(p, 0.0);
    for (int i=0;i<n;++i)
        for (int j=0;j<p;++j)
            overall_mean[j]+=X[i][j];
    for (int j=0;j<p;++j) overall_mean[j]/=n;

    for (int i=0;i<n;++i) {
        int ci=(int)(std::find(classes.begin(), classes.end(), y[i])-classes.begin());
        counts[ci]++;
        for (int j=0;j<p;++j) mean[ci][j]+=X[i][j];
    }
    for (int c=0;c<C;++c) {
        class_prior[c]=(double)counts[c]/n;
        if (counts[c]>0)
            for (int j=0;j<p;++j) mean[c][j]/=counts[c];
    }

    Mat Sw=mat_zeros(p, p);
    for (int i=0;i<n;++i) {
        int ci=(int)(std::find(classes.begin(), classes.end(), y[i])-classes.begin());
        Vec d=vec_sub(X[i], mean[ci]);
        for (int a=0;a<p;++a)
            for (int b=0;b<p;++b)
                Sw[a][b]+=d[a]*d[b];
    }
    double dof=std::max(1.0, (double)(n-C));
    for (int a=0;a<p;++a)
        for (int b=0;b<p;++b)
            Sw[a][b]/=dof;

    pooled_cov_inv=mat_inv(Sw, reg_epsilon);

    discrim_const.assign(C, 0.0);
    discrim_coef.assign(C, Vec(p, 0.0));
    for (int c=0;c<C;++c) {
        Vec Smu=mat_vec(pooled_cov_inv, mean[c]);
        discrim_coef[c]=Smu;
        discrim_const[c]=-0.5*vec_dot(mean[c], Smu)+std::log(std::max(class_prior[c], 1e-300));
    }

    int k_comp=n_components>0 ? n_components : std::min(C-1, p);
    if (k_comp>0) {
        Mat Sb=mat_zeros(p, p);
        for (int c=0;c<C;++c) {
            Vec d=vec_sub(mean[c], overall_mean);
            for (int a=0;a<p;++a)
                for (int b=0;b<p;++b)
                    Sb[a][b]+=counts[c]*d[a]*d[b];
        }
        Mat Sw_inv=mat_inv(Sw, reg_epsilon);
        Mat M=mat_mul(Sw_inv, Sb);
        Mat evecs; Vec evals;
        sym_eig_power(M, evecs, evals, k_comp);
        projection=evecs;
    }
}

Vec LDA::predict(const Mat& X) const {
    Vec pred(X.size(), classes.empty()?0.0:classes[0]);
    if (classes.empty()||discrim_coef.empty()) return pred;
    int C=(int)classes.size();
    for (size_t i=0;i<X.size();++i) {
        double best=-1e300; int best_c=0;
        for (int c=0;c<C;++c) {
            double score=discrim_const[c]+vec_dot(discrim_coef[c], X[i]);
            if (score>best) { best=score; best_c=c; }
        }
        pred[i]=classes[best_c];
    }
    return pred;
}

double LDA::score(const Mat& X, const Vec& y) const { return accuracy(predict(X), y); }

Mat LDA::transform(const Mat& X) const {
    if (projection.empty()||overall_mean.empty()) return {};
    int k=(int)projection.size(), p=(int)overall_mean.size();
    Mat Z(X.size(), Vec(k, 0.0));
    for (size_t i=0;i<X.size();++i) {
        Vec xc(p);
        for (int j=0;j<p;++j) xc[j]=X[i][j]-overall_mean[j];
        for (int r=0;r<k;++r) Z[i][r]=vec_dot(projection[r], xc);
    }
    return Z;
}

// ========================== Quadratic Discriminant Analysis ==========================

void QDA::fit(const Mat& X, const Vec& y) {
    classes.clear(); class_prior.clear(); mean.clear();
    quad_coef.clear(); linear_coef.clear(); discrim_const.clear();
    if (X.empty()) return;

    std::set<double> cls(y.begin(), y.end());
    classes.assign(cls.begin(), cls.end());
    int C=(int)classes.size(), n=(int)X.size(), p=(int)X[0].size();
    if (C<1) return;

    mean.assign(C, Vec(p, 0.0));
    class_prior.assign(C, 0.0);
    std::vector<int> counts(C, 0);
    std::vector<Mat> class_X(C);

    for (int i=0;i<n;++i) {
        int ci=(int)(std::find(classes.begin(), classes.end(), y[i])-classes.begin());
        counts[ci]++;
        class_X[ci].push_back(X[i]);
        for (int j=0;j<p;++j) mean[ci][j]+=X[i][j];
    }
    for (int c=0;c<C;++c) {
        class_prior[c]=(double)counts[c]/n;
        if (counts[c]>0)
            for (int j=0;j<p;++j) mean[c][j]/=counts[c];
    }

    quad_coef.assign(C, Vec(p*p, 0.0));
    linear_coef.assign(C, Vec(p, 0.0));
    discrim_const.assign(C, 0.0);

    for (int c=0;c<C;++c) {
        Mat cov=mat_zeros(p, p);
        if (counts[c]>1) {
            cov=class_scatter(class_X[c], mean[c]);
            double denom=std::max(1.0, (double)(counts[c]-1));
            for (int a=0;a<p;++a)
                for (int b=0;b<p;++b)
                    cov[a][b]/=denom;
        } else {
            for (int a=0;a<p;++a) cov[a][a]=1.0;
        }
        Mat inv_cov=mat_inv(cov, reg_epsilon);
        double log_det=mat_logdet_spd(cov, reg_epsilon);
        Vec inv_mu=mat_vec(inv_cov, mean[c]);
        double mu_quad=vec_dot(mean[c], inv_mu);

        for (int a=0;a<p;++a)
            for (int b=0;b<p;++b)
                quad_coef[c][a*p+b]=inv_cov[a][b];
        linear_coef[c]=inv_mu;
        discrim_const[c]=-0.5*log_det-0.5*mu_quad+std::log(std::max(class_prior[c], 1e-300));
    }
}

Vec QDA::predict(const Mat& X) const {
    Vec pred(X.size(), classes.empty()?0.0:classes[0]);
    if (classes.empty()||quad_coef.empty()) return pred;
    int C=(int)classes.size(), p=(int)X[0].size();
    for (size_t i=0;i<X.size();++i) {
        double best=-1e300; int best_c=0;
        for (int c=0;c<C;++c) {
            Mat inv_cov(p, Vec(p));
            for (int a=0;a<p;++a)
                for (int b=0;b<p;++b)
                    inv_cov[a][b]=quad_coef[c][a*p+b];
            double score=discrim_const[c]+vec_dot(linear_coef[c], X[i])
                -0.5*mat_quad_form(inv_cov, X[i]);
            if (score>best) { best=score; best_c=c; }
        }
        pred[i]=classes[best_c];
    }
    return pred;
}

double QDA::score(const Mat& X, const Vec& y) const { return accuracy(predict(X), y); }

// ========================== Decision Tree ==========================

static double tree_idx_weight_sum(const Vec& weights, const std::vector<int>& idx) {
    if (weights.empty()) return (double)idx.size();
    double s = 0.0;
    for (int i : idx) s += weights[(size_t)i];
    return s;
}

static double gini(const Vec& y, const std::vector<int>& idx, const Vec& weights = {}) {
    if (idx.empty()) return 0;
    double tot = tree_idx_weight_sum(weights, idx);
    if (tot <= 0.0) return 0;
    std::map<double, double> cnt;
    for (int i : idx) {
        double w = weights.empty() ? 1.0 : weights[(size_t)i];
        cnt[y[i]] += w;
    }
    double g = 1.0;
    for (auto& [k, v] : cnt) {
        double p = v / tot;
        g -= p * p;
    }
    return g;
}

static double mse_impurity(const Vec& y, const std::vector<int>& idx, const Vec& weights = {}) {
    if (idx.empty()) return 0;
    double tot = tree_idx_weight_sum(weights, idx);
    if (tot <= 0.0) return 0;
    double mean = 0;
    for (int i : idx) {
        double w = weights.empty() ? 1.0 : weights[(size_t)i];
        mean += w * y[i];
    }
    mean /= tot;
    double s = 0;
    for (int i : idx) {
        double w = weights.empty() ? 1.0 : weights[(size_t)i];
        double d = y[i] - mean;
        s += w * d * d;
    }
    return s / tot;
}

static double tree_impurity(const Vec& y, const std::vector<int>& idx,
                            const std::string& criterion, const Vec& weights = {}) {
    return criterion == "mse" ? mse_impurity(y, idx, weights) : gini(y, idx, weights);
}

static double tree_leaf_value(const Vec& y, const std::vector<int>& idx,
                              const std::string& criterion, const Vec& weights = {}) {
    if (idx.empty()) return 0;
    if (criterion == "mse") {
        double tot = tree_idx_weight_sum(weights, idx);
        if (tot <= 0.0) return 0;
        double s = 0;
        for (int i : idx) {
            double w = weights.empty() ? 1.0 : weights[(size_t)i];
            s += w * y[i];
        }
        return s / tot;
    }
    std::map<double, double> cnt;
    for (int i : idx) {
        double w = weights.empty() ? 1.0 : weights[(size_t)i];
        cnt[y[i]] += w;
    }
    return std::max_element(cnt.begin(), cnt.end(),
                            [](auto& a, auto& b) { return a.second < b.second; })->first;
}

static bool tree_all_same(const Vec& y, const std::vector<int>& idx, const std::string& criterion) {
    if (idx.size() <= 1) return true;
    if (criterion == "mse") {
        double ref = y[idx[0]];
        for (int i : idx) if (std::abs(y[i] - ref) > 1e-12) return false;
        return true;
    }
    for (int i : idx) if (y[i] != y[idx[0]]) return false;
    return true;
}

int DecisionTree::build(const Mat& X, const Vec& y, std::vector<int>& idx, int depth) {
    const Vec& weights = fit_weights_;
    int node_idx = (int)nodes.size();
    nodes.push_back({});
    if (depth >= max_depth || idx.size() <= 1) {
        nodes[node_idx].value = tree_leaf_value(y, idx, criterion, weights);
        return node_idx;
    }
    if (tree_all_same(y, idx, criterion)) {
        nodes[node_idx].value = tree_leaf_value(y, idx, criterion, weights);
        return node_idx;
    }

    double best_g = 1e300;
    int best_f = -1;
    double best_t = 0;
    int p = (int)X[0].size();
    double idx_w = tree_idx_weight_sum(weights, idx);
    for (int f = 0; f < p; ++f) {
        std::vector<double> vals;
        for (int i : idx) vals.push_back(X[i][f]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        for (size_t vi = 0; vi + 1 < vals.size(); ++vi) {
            double t = (vals[vi] + vals[vi + 1]) / 2;
            std::vector<int> li, ri;
            for (int i : idx) (X[i][f] <= t ? li : ri).push_back(i);
            if (li.empty() || ri.empty()) continue;
            double lw = tree_idx_weight_sum(weights, li);
            double rw = tree_idx_weight_sum(weights, ri);
            double g = (lw * tree_impurity(y, li, criterion, weights)
                        + rw * tree_impurity(y, ri, criterion, weights)) / idx_w;
            if (g < best_g) { best_g = g; best_f = f; best_t = t; }
        }
    }
    if (best_f < 0) {
        nodes[node_idx].value = tree_leaf_value(y, idx, criterion, weights);
        return node_idx;
    }
    nodes[node_idx].feature = best_f;
    nodes[node_idx].threshold = best_t;
    std::vector<int> li, ri;
    for (int i : idx) (X[i][best_f] <= best_t ? li : ri).push_back(i);
    nodes[node_idx].left = build(X, y, li, depth + 1);
    nodes[node_idx].right = build(X, y, ri, depth + 1);
    return node_idx;
}

void DecisionTree::fit(const Mat& X, const Vec& y) {
    fit_weights_.clear();
    nodes.clear();
    std::vector<int> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0);
    build(X, y, idx, 0);
}

void DecisionTree::fit(const Mat& X, const Vec& y, const Vec& sample_weights) {
    fit_weights_.clear();
    if (!sample_weights.empty() && sample_weights.size() == X.size())
        fit_weights_ = sample_weights;
    nodes.clear();
    std::vector<int> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0);
    build(X, y, idx, 0);
    fit_weights_.clear();
}

double DecisionTree::predict_one(const Vec& x, int n) const {
    const auto& node=nodes[n];
    if (node.feature<0) return node.value;
    return predict_one(x, x[node.feature]<=node.threshold ? node.left : node.right);
}
Vec DecisionTree::predict(const Mat& X) const {
    Vec p(X.size());
    for (size_t i=0;i<X.size();++i) p[i]=predict_one(X[i],0);
    return p;
}
double DecisionTree::score(const Mat& X, const Vec& y) const {
    return criterion=="mse" ? r2_score(predict(X),y) : accuracy(predict(X),y);
}

// ========================== Random Forest ==========================

static Mat select_columns(const Mat& X, const std::vector<int>& cols) {
    Mat out(X.size(), Vec(cols.size()));
    for (size_t i=0;i<X.size();++i)
        for (size_t j=0;j<cols.size();++j)
            out[i][j]=X[i][cols[j]];
    return out;
}

void RandomForest::fit(const Mat& X, const Vec& y) {
    trees.clear();
    feature_indices.clear();
    int n=(int)X.size(), p=(int)X[0].size();
    std::mt19937 rng(config.seed);
    std::uniform_int_distribution<int> row_dist(0, std::max(0,n-1));

    size_t n_feat = std::max<size_t>(1, (size_t)std::round(config.feature_subsample_ratio * p));
    size_t n_sample = std::max<size_t>(1, (size_t)std::round(config.sample_subsample_ratio * n));

    for (size_t t=0;t<config.n_trees;++t) {
        std::vector<int> all_feat(p);
        std::iota(all_feat.begin(), all_feat.end(), 0);
        std::shuffle(all_feat.begin(), all_feat.end(), rng);
        std::vector<int> feat(all_feat.begin(), all_feat.begin()+(int)n_feat);
        std::sort(feat.begin(), feat.end());

        Mat X_boot; Vec y_boot;
        X_boot.reserve(n_sample);
        y_boot.reserve(n_sample);
        for (size_t s=0;s<n_sample;++s) {
            int idx=row_dist(rng);
            X_boot.push_back(X[idx]);
            y_boot.push_back(y[idx]);
        }

        Mat X_sub = select_columns(X_boot, feat);
        DecisionTree tree((int)config.max_depth, "gini");
        tree.fit(X_sub, y_boot);
        trees.push_back(std::move(tree));
        feature_indices.push_back(std::move(feat));
    }
}

Vec RandomForest::predict(const Mat& X) const {
    Vec out(X.size());
    for (size_t i=0;i<X.size();++i) {
        std::map<double,int> votes;
        for (size_t t=0;t<trees.size();++t) {
            Mat X_sub = select_columns(X, feature_indices[t]);
            votes[trees[t].predict(X_sub)[i]]++;
        }
        out[i]=std::max_element(votes.begin(),votes.end(),[](auto&a,auto&b){return a.second<b.second;})->first;
    }
    return out;
}

// ========================== Gradient Boosting ==========================

void GradientBoosting::fit(const Mat& X, const Vec& y) {
    trees.clear();
    init_prediction=0;
    for (double v:y) init_prediction+=v;
    init_prediction/=y.size();

    Vec current(y.size(), init_prediction);
    for (size_t t=0;t<config.n_trees;++t) {
        Vec residuals = vec_sub(y, current);
        DecisionTree tree((int)config.max_depth, "mse");
        tree.fit(X, residuals);
        Vec tree_pred = tree.predict(X);
        for (size_t i=0;i<current.size();++i)
            current[i]+=config.learning_rate*tree_pred[i];
        trees.push_back(std::move(tree));
    }
    (void)config.seed;
}

Vec GradientBoosting::predict(const Mat& X) const {
    Vec pred(X.size(), init_prediction);
    for (const auto& tree : trees) {
        Vec tree_pred = tree.predict(X);
        for (size_t i=0;i<pred.size();++i)
            pred[i]+=config.learning_rate*tree_pred[i];
    }
    return pred;
}

// ========================== AdaBoost (SAMME) ==========================

static bool adaboost_binary_labels(const Vec& y) {
    for (double v : y)
        if (v != 0.0 && v != 1.0) return false;
    return true;
}

void AdaBoost::fit(const Mat& X, const Vec& y) {
    estimators.clear();
    estimator_weights.clear();
    if (X.empty() || y.empty() || X.size() != y.size() || !adaboost_binary_labels(y))
        return;

    int n = (int)X.size();
    Vec weights((size_t)n, 1.0 / n);

    for (size_t m = 0; m < config.n_estimators; ++m) {
        DecisionTree stump((int)config.max_depth, "gini");
        stump.fit(X, y, weights);
        Vec preds = stump.predict(X);

        double err = 0.0;
        for (int i = 0; i < n; ++i)
            if (std::abs(preds[(size_t)i] - y[(size_t)i]) >= 0.5)
                err += weights[(size_t)i];
        err = std::clamp(err, 1e-10, 1.0 - 1e-10);

        if (err >= 0.5 - 1e-12) break;

        // SAMME: alpha = log((1-err)/err) + log(K-1); K=2 => second term is 0.
        double alpha = std::log((1.0 - err) / err);

        estimators.push_back(std::move(stump));
        estimator_weights.push_back(alpha);

        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(preds[(size_t)i] - y[(size_t)i]) >= 0.5)
                weights[(size_t)i] *= std::exp(alpha);
            norm += weights[(size_t)i];
        }
        if (norm <= 1e-14) break;
        for (int i = 0; i < n; ++i) weights[(size_t)i] /= norm;
    }
}

Vec AdaBoost::predict(const Mat& X) const {
    Vec pred(X.size(), 0.0);
    if (estimators.empty()) return pred;

    for (size_t i = 0; i < X.size(); ++i) {
        double vote0 = 0.0, vote1 = 0.0;
        for (size_t m = 0; m < estimators.size(); ++m) {
            double h = estimators[m].predict({X[i]})[0];
            if (h >= 0.5) vote1 += estimator_weights[m];
            else vote0 += estimator_weights[m];
        }
        pred[i] = vote1 >= vote0 ? 1.0 : 0.0;
    }
    return pred;
}

double AdaBoost::score(const Mat& X, const Vec& y) const {
    return accuracy(predict(X), y);
}

// ========================== Support Vector Machine (SMO) ==========================

static bool svm_normalize_labels(Vec& y) {
    bool saw0=false, saw_neg1=false;
    for (double v:y) {
        if (v==0.0) saw0=true;
        else if (v==1.0) {}
        else if (v==-1.0) saw_neg1=true;
        else return false;
    }
    if (saw0 && saw_neg1) return false;
    if (saw0)
        for (auto& v:y) v=v==0.0?-1.0:1.0;
    return true;
}

static double svm_linear_kernel(const Vec& xi, const Vec& xj) {
    return vec_dot(xi, xj);
}

static double svm_rbf_kernel(const Vec& xi, const Vec& xj, double gamma) {
    double sq=0;
    for (size_t k=0;k<xi.size();++k) {
        double d=xi[k]-xj[k];
        sq+=d*d;
    }
    return std::exp(-gamma*sq);
}

static double svm_kernel(const Vec& xi, const Vec& xj, const SVMConfig& cfg) {
    return cfg.kernel==SVMKernel::Linear
        ? svm_linear_kernel(xi, xj)
        : svm_rbf_kernel(xi, xj, cfg.gamma);
}

void SVM::fit(const Mat& X, const Vec& y_in) {
    support_vectors.clear();
    alphas.clear();
    sv_labels.clear();
    b=0.0;
    if (X.empty() || y_in.empty() || X.size()!=y_in.size()) return;

    Vec y=y_in;
    if (!svm_normalize_labels(y)) return;

    int n=(int)X.size();
    double C=config.C, tol=config.tol;

    Mat K(n, Vec(n));
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            K[i][j]=svm_kernel(X[i], X[j], config);

    Vec alpha(n, 0.0);
    Vec E(n);
    for (int i=0;i<n;++i) E[i]=-y[i];

    int passes=0;
    while (passes<config.max_iter) {
        int num_changed=0;
        for (int i=0;i<n;++i) {
            double fi=b;
            for (int j=0;j<n;++j) fi+=alpha[j]*y[j]*K[i][j];
            E[i]=fi-y[i];

            double ri=y[i]*fi;
            bool kkt_violated =
                (alpha[i]<tol && ri<1.0-tol) ||
                (alpha[i]>C-tol && ri>1.0+tol) ||
                (alpha[i]>=tol && alpha[i]<=C-tol && std::abs(ri-1.0)>tol);
            if (!kkt_violated) continue;

            int j=-1;
            double max_delta=0;
            for (int jj=0;jj<n;++jj) {
                if (jj==i) continue;
                double delta=std::abs(E[i]-E[jj]);
                if (delta>max_delta) { max_delta=delta; j=jj; }
            }
            if (j<0) continue;

            double eta=K[i][i]+K[j][j]-2.0*K[i][j];
            if (eta<=1e-12) continue;

            double ai=alpha[i], aj=alpha[j];
            double Ei=E[i], Ej=E[j];

            double L, H;
            if (y[i]!=y[j]) {
                L=std::max(0.0, aj-ai);
                H=std::min(C, C+aj-ai);
            } else {
                L=std::max(0.0, ai+aj-C);
                H=std::min(C, ai+aj);
            }
            if (L>=H) continue;

            double aj_new=aj+y[j]*(Ei-Ej)/eta;
            aj_new=std::clamp(aj_new, L, H);
            if (std::abs(aj_new-aj)<1e-12) continue;

            double ai_new=ai+y[i]*y[j]*(aj-aj_new);

            double b1=b-Ei-y[i]*(ai_new-ai)*K[i][i]-y[j]*(aj_new-aj)*K[i][j];
            double b2=b-Ej-y[i]*(ai_new-ai)*K[i][j]-y[j]*(aj_new-aj)*K[j][j];

            if (ai_new>tol && ai_new<C-tol) b=b1;
            else if (aj_new>tol && aj_new<C-tol) b=b2;
            else b=(b1+b2)*0.5;

            double dai=ai_new-ai, daj=aj_new-aj;
            alpha[i]=ai_new;
            alpha[j]=aj_new;

            for (int k=0;k<n;++k)
                E[k]+=y[i]*dai*K[i][k]+y[j]*daj*K[j][k];

            num_changed++;
        }
        if (num_changed==0) ++passes;
        else passes=0;
    }

    this->b=b;
    for (int i=0;i<n;++i) {
        if (alpha[i]>1e-7) {
            support_vectors.push_back(X[i]);
            alphas.push_back(alpha[i]);
            sv_labels.push_back(y[i]);
        }
    }
}

Vec SVM::decision_function(const Mat& X) const {
    Vec out(X.size(), b);
    for (size_t i=0;i<X.size();++i) {
        double sum=b;
        for (size_t j=0;j<support_vectors.size();++j)
            sum+=alphas[j]*sv_labels[j]*svm_kernel(support_vectors[j], X[i], config);
        out[i]=sum;
    }
    return out;
}

Vec SVM::predict(const Mat& X) const {
    auto df=decision_function(X);
    for (auto& v:df) v=v>=0.0?1.0:-1.0;
    return df;
}

// ========================== KMeans ==========================

void KMeans::fit(const Mat& X) {
    int n=X.size(), p=X[0].size();
    // K-means++ init
    std::mt19937 rng(42);
    centers.clear();
    centers.reserve(static_cast<size_t>(k));
    std::uniform_int_distribution<int> uid(0,n-1);
    centers.push_back(X[uid(rng)]);
    Vec dists(n);
    for (int ci=1;ci<k;++ci) {
        for (int i=0;i<n;++i) {
            double best=1e300;
            for (const auto& c:centers) best=std::min(best,sq_eucl_dist(X[i],c));
            dists[i]=best;
        }
        double tot=0; for (double d:dists) tot+=d;
        std::uniform_real_distribution<double> ud(0,tot);
        double r=ud(rng); double cum=0;
        for (int i=0;i<n;++i) { cum+=dists[i]; if (cum>=r){centers.push_back(X[i]);break;} }
        if ((int)centers.size()<ci+1) centers.push_back(X[n-1]);
    }

    labels_.resize(static_cast<size_t>(n));
    Mat new_centers(k, Vec(p,0.0));
    std::vector<int> cnt(static_cast<size_t>(k),0);
    for (int iter=0;iter<max_iter;++iter) {
        // Assign
        bool changed=false;
        for (int i=0;i<n;++i) {
            double best=1e300; int bi=0;
            for (int c=0;c<k;++c) {
                const double d=sq_eucl_dist(X[i],centers[c]);
                if (d<best){best=d;bi=c;}
            }
            if (labels_[i]!=bi) changed=true;
            labels_[i]=bi;
        }
        // Update (reuse accumulation buffers)
        for (int c=0;c<k;++c) {
            cnt[c]=0;
            std::fill(new_centers[c].begin(), new_centers[c].end(), 0.0);
        }
        for (int i=0;i<n;++i) {
            const int lab=labels_[i];
            cnt[lab]++;
            for (int j=0;j<p;++j) new_centers[lab][j]+=X[i][j];
        }
        for (int c=0;c<k;++c)
            if (cnt[c]>0)
                for (int j=0;j<p;++j) new_centers[c][j]/=cnt[c];
        centers.swap(new_centers);
        if (!changed) break;
    }
}
Vec KMeans::predict(const Mat& X) const {
    Vec pred(X.size());
    for (size_t i=0;i<X.size();++i) {
        double best=1e300; int bi=0;
        for (int c=0;c<k;++c) {
            const double d=sq_eucl_dist(X[i],centers[c]);
            if (d<best){best=d;bi=c;}
        }
        pred[i]=bi;
    }
    return pred;
}
double KMeans::inertia(const Mat& X) const {
    double s=0;
    for (size_t i=0;i<X.size();++i) { double d=eucl_dist(X[i],centers[(int)labels_[i]]); s+=d*d; }
    return s;
}

// ========================== Gaussian Mixture Model ==========================

static constexpr double GMM_VAR_FLOOR = 1e-6;

static double gmm_log_gaussian_diag(const Vec& x, const Vec& mean, const Vec& var) {
    int p=(int)x.size();
    double log_det=0, mahal=0;
    for (int j=0;j<p;++j) {
        double v=std::max(var[j], GMM_VAR_FLOOR);
        double d=x[j]-mean[j];
        mahal+=d*d/v;
        log_det+=std::log(v);
    }
    return -0.5*(p*std::log(2*M_PI)+log_det+mahal);
}

static Mat gmm_responsibilities(const Mat& X, const Mat& means, const Mat& variances,
                                const Vec& weights, double* log_likelihood=nullptr) {
    int n=(int)X.size(), p=(int)X[0].size();
    size_t k=means.size();
    Mat resp(n, Vec(k, 0.0));
    double ll=0;
    for (int i=0;i<n;++i) {
        Vec log_prob(k);
        for (size_t c=0;c<k;++c) {
            log_prob[c]=std::log(std::max(weights[c], 1e-300))
                +gmm_log_gaussian_diag(X[i], means[c], variances[c]);
        }
        double max_lp=*std::max_element(log_prob.begin(), log_prob.end());
        double sum_exp=0;
        for (size_t c=0;c<k;++c) {
            resp[i][c]=std::exp(log_prob[c]-max_lp);
            sum_exp+=resp[i][c];
        }
        for (size_t c=0;c<k;++c) resp[i][c]/=sum_exp;
        ll+=max_lp+std::log(sum_exp);
    }
    if (log_likelihood) *log_likelihood=ll;
    return resp;
}

void GaussianMixture::fit(const Mat& X) {
    means.clear(); variances.clear(); weights.clear();
    log_likelihood=0.0;
    if (X.empty()) return;
    int n=(int)X.size(), p=(int)X[0].size();
    size_t k=std::min(config.n_components, (size_t)n);
    if (k==0) return;

    std::mt19937 rng(config.seed);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    means.resize(k, Vec(p));
    for (size_t c=0;c<k;++c) means[c]=X[idx[c]];

    Vec global_mean(p, 0.0);
    for (int i=0;i<n;++i) for (int j=0;j<p;++j) global_mean[j]+=X[i][j];
    for (int j=0;j<p;++j) global_mean[j]/=n;

    Vec global_var(p, GMM_VAR_FLOOR);
    for (int i=0;i<n;++i)
        for (int j=0;j<p;++j) {
            double d=X[i][j]-global_mean[j];
            global_var[j]+=d*d;
        }
    for (int j=0;j<p;++j) global_var[j]=std::max(global_var[j]/n, GMM_VAR_FLOOR);

    variances.assign(k, global_var);
    weights.assign(k, 1.0/(double)k);

    double prev_ll=-std::numeric_limits<double>::infinity();
    for (size_t iter=0; iter<config.max_iter; ++iter) {
        Mat resp=gmm_responsibilities(X, means, variances, weights, &log_likelihood);
        if (iter>0 && log_likelihood-prev_ll<config.tol) break;
        prev_ll=log_likelihood;

        Vec nk(k, 0.0);
        for (int i=0;i<n;++i) for (size_t c=0;c<k;++c) nk[c]+=resp[i][c];

        for (size_t c=0;c<k;++c) {
            double denom=std::max(nk[c], 1e-12);
            weights[c]=nk[c]/n;
            for (int j=0;j<p;++j) {
                double m=0;
                for (int i=0;i<n;++i) m+=resp[i][c]*X[i][j];
                means[c][j]=m/denom;
            }
            for (int j=0;j<p;++j) {
                double v=0;
                for (int i=0;i<n;++i) {
                    double d=X[i][j]-means[c][j];
                    v+=resp[i][c]*d*d;
                }
                variances[c][j]=std::max(v/denom, GMM_VAR_FLOOR);
            }
        }
    }
}

Mat GaussianMixture::predict_proba(const Mat& X) const {
    if (X.empty()||means.empty()) return {};
    return gmm_responsibilities(X, means, variances, weights, nullptr);
}

Vec GaussianMixture::predict(const Mat& X) const {
    Vec pred(X.size());
    if (X.empty()||means.empty()) return pred;
    Mat proba=predict_proba(X);
    for (size_t i=0;i<X.size();++i) {
        int bi=0;
        for (size_t c=1;c<proba[i].size();++c)
            if (proba[i][c]>proba[i][bi]) bi=(int)c;
        pred[i]=bi;
    }
    return pred;
}

double GaussianMixture::score(const Mat& X) const {
    if (X.empty()||means.empty()) return -std::numeric_limits<double>::infinity();
    double ll=0;
    gmm_responsibilities(X, means, variances, weights, &ll);
    return ll/X.size();
}

// ========================== Isolation Forest ==========================

double IsolationForest::avg_path_length(size_t n) {
    if (n <= 1) return 0.0;
    if (n == 2) return 1.0;
    double h = 0.0;
    for (size_t i = 1; i <= n - 1; ++i) h += 1.0 / (double)i;
    return 2.0 * h - 2.0 * (double)(n - 1) / (double)n;
}

int IsolationForest::build_tree(Tree& tree, const Mat& X, const std::vector<int>& idx,
                                std::mt19937& rng, int depth, int n_feat) {
    int node_idx = (int)tree.nodes.size();
    tree.nodes.push_back({});
    tree.nodes[node_idx].size = idx.size();

    if (depth >= (int)tree.height_limit || idx.size() <= 1) return node_idx;

    std::uniform_int_distribution<int> feat_dist(0, std::max(0, n_feat - 1));
    int feat = feat_dist(rng);

    double lo = X[idx[0]][feat], hi = lo;
    for (int i : idx) {
        lo = std::min(lo, X[i][feat]);
        hi = std::max(hi, X[i][feat]);
    }
    if (lo >= hi) return node_idx;

    std::uniform_real_distribution<double> thresh_dist(lo, hi);
    double thresh = thresh_dist(rng);

    std::vector<int> left_idx, right_idx;
    left_idx.reserve(idx.size());
    right_idx.reserve(idx.size());
    for (int i : idx) {
        if (X[i][feat] < thresh) left_idx.push_back(i);
        else right_idx.push_back(i);
    }
    if (left_idx.empty() || right_idx.empty()) return node_idx;

    tree.nodes[node_idx].feature = feat;
    tree.nodes[node_idx].threshold = thresh;
    tree.nodes[node_idx].left = build_tree(tree, X, left_idx, rng, depth + 1, n_feat);
    tree.nodes[node_idx].right = build_tree(tree, X, right_idx, rng, depth + 1, n_feat);
    return node_idx;
}

double IsolationForest::path_length_one(const Tree& tree, const Vec& x, int node) {
    if (node < 0 || node >= (int)tree.nodes.size()) return 0.0;
    const auto& nd = tree.nodes[node];
    if (nd.feature < 0) {
        return nd.size > 1 ? avg_path_length(nd.size) : 0.0;
    }
    double edge = 1.0;
    if (x[nd.feature] < nd.threshold)
        return edge + path_length_one(tree, x, nd.left);
    return edge + path_length_one(tree, x, nd.right);
}

void IsolationForest::fit(const Mat& X) {
    trees_.clear();
    subsample_size_ = 0;
    n_features_ = 0;
    if (X.empty() || X[0].empty()) return;

    int n = (int)X.size();
    int p = (int)X[0].size();
    n_features_ = p;
    subsample_size_ = std::min(sample_size, (size_t)n);
    if (subsample_size_ == 0) return;

    size_t height_limit = subsample_size_ <= 1 ? 0
        : (size_t)std::ceil(std::log2((double)subsample_size_));

    std::mt19937 rng(seed);
    std::vector<int> all_idx(n);
    std::iota(all_idx.begin(), all_idx.end(), 0);

    trees_.reserve(n_trees);
    for (size_t t = 0; t < n_trees; ++t) {
        std::shuffle(all_idx.begin(), all_idx.end(), rng);
        std::vector<int> sample_idx(all_idx.begin(),
                                    all_idx.begin() + (int)subsample_size_);

        Tree tree;
        tree.height_limit = height_limit;
        build_tree(tree, X, sample_idx, rng, 0, p);
        trees_.push_back(std::move(tree));
    }
}

double IsolationForest::raw_score(const Vec& x) const {
    if (trees_.empty() || x.size() != (size_t)n_features_) return 0.5;
    double c = avg_path_length(subsample_size_);
    if (c <= 0.0) return 0.5;

    double sum = 0.0;
    for (const auto& tree : trees_)
        sum += path_length_one(tree, x, 0);
    double eh = sum / (double)trees_.size();
    return std::pow(2.0, -eh / c);
}

double IsolationForest::anomaly_score(const Vec& x) const {
    return raw_score(x);
}

Vec IsolationForest::anomaly_scores(const Mat& X) const {
    Vec scores(X.size(), 0.5);
    for (size_t i = 0; i < X.size(); ++i) scores[i] = raw_score(X[i]);
    return scores;
}

// ========================== DBSCAN ==========================

void DBSCAN::fit(const Mat& X) {
    int n=X.size();
    labels_.assign(n,-1);
    int cluster=-1;
    std::vector<bool> visited(n,false);

    std::function<std::vector<int>(int)> neighbors=[&](int i)->std::vector<int>{
        std::vector<int> nb;
        for (int j=0;j<n;++j) if (j!=i && eucl_dist(X[i],X[j])<=eps) nb.push_back(j);
        return nb;
    };

    for (int i=0;i<n;++i) {
        if (visited[i]) continue;
        visited[i]=true;
        auto nb=neighbors(i);
        if ((int)nb.size()<min_samples) { labels_[i]=-1; continue; }
        labels_[i]=++cluster;
        std::vector<int> seeds=nb;
        for (size_t si=0;si<seeds.size();++si) {
            int q=seeds[si];
            if (!visited[q]) {
                visited[q]=true;
                auto nb2=neighbors(q);
                if ((int)nb2.size()>=min_samples) seeds.insert(seeds.end(),nb2.begin(),nb2.end());
            }
            if (labels_[q]==-1) labels_[q]=cluster;
        }
    }
}

// ========================== Spectral Clustering ==========================

static ms::ColMatrix<double> ml_mat_to_col(const Mat& M) {
    size_t rows = M.size(), cols = M.empty() ? 0 : M[0].size();
    ms::ColMatrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out(i, j) = M[i][j];
    return out;
}

static double rbf_affinity(double sq_dist, double inv_two_sigma_sq) {
    return std::exp(-sq_dist * inv_two_sigma_sq);
}

static Mat build_affinity(const Mat& X, double sigma, int n_neighbors) {
    int n = (int)X.size();
    Mat W(n, Vec(n, 0.0));
    if (n == 0) return W;

    double sig = sigma > 0.0 ? sigma : 1.0;
    double inv_two_sigma_sq = 1.0 / (2.0 * sig * sig);

    if (n_neighbors <= 0) {
        for (int i = 0; i < n; ++i) {
            W[i][i] = 1.0;
            for (int j = i + 1; j < n; ++j) {
                double w = rbf_affinity(sq_eucl_dist(X[i], X[j]), inv_two_sigma_sq);
                W[i][j] = W[j][i] = w;
            }
        }
        return W;
    }

    int k = std::min(n_neighbors, n - 1);
    if (k <= 0) {
        for (int i = 0; i < n; ++i) W[i][i] = 1.0;
        return W;
    }

    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> dists;
        dists.reserve((size_t)(n - 1));
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            dists.push_back({sq_eucl_dist(X[i], X[j]), j});
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        for (int t = 0; t < k; ++t) {
            int j = dists[(size_t)t].second;
            double w = rbf_affinity(dists[(size_t)t].first, inv_two_sigma_sq);
            W[i][j] = std::max(W[i][j], w);
            W[j][i] = std::max(W[j][i], w);
        }
        W[i][i] = 1.0;
    }
    return W;
}

static Mat symmetric_normalized_laplacian(const Mat& W) {
    int n = (int)W.size();
    Mat L = mat_eye(n);
    if (n == 0) return L;

    Vec deg(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) deg[i] += W[i][j];
    }

    for (int i = 0; i < n; ++i) {
        double inv_sqrt_d = deg[i] > 1e-14 ? 1.0 / std::sqrt(deg[i]) : 0.0;
        for (int j = 0; j < n; ++j) {
            double inv_sqrt_e = deg[j] > 1e-14 ? 1.0 / std::sqrt(deg[j]) : 0.0;
            L[i][j] -= inv_sqrt_d * W[i][j] * inv_sqrt_e;
        }
    }

    // Numerical symmetrization for eig_sym's symmetry check.
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            L[i][j] = L[j][i] = 0.5 * (L[i][j] + L[j][i]);
    return L;
}

static ms::ColMatrix<double> ml_jacobi_eigen_symmetric(ms::ColMatrix<double> A,
                                                        ms::ColMatrix<double>& V,
                                                        double tol = 1e-12) {
    const size_t n = A.rows();
    V = ms::ColMatrix<double>(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) V(i, i) = 1.0;

    for (int sweep = 0; sweep < 100; ++sweep) {
        double max_off = 0.0;
        size_t p = 0, q = 1;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(A(i, j)) > max_off) {
                    max_off = std::abs(A(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol) break;

        const double app = A(p, p), aqq = A(q, q), apq = A(p, q);
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi), s = std::sin(phi);

        for (size_t k = 0; k < n; ++k) {
            const double akp = A(k, p), akq = A(k, q);
            A(k, p) = c * akp - s * akq;
            A(p, k) = A(k, p);
            A(k, q) = s * akp + c * akq;
            A(q, k) = A(k, q);
        }
        A(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A(p, q) = A(q, p) = 0.0;

        for (size_t k = 0; k < n; ++k) {
            const double vkp = V(k, p), vkq = V(k, q);
            V(k, p) = c * vkp - s * vkq;
            V(k, q) = s * vkp + c * vkq;
        }
    }

    ms::ColMatrix<double> values(n, 1, 0.0);
    for (size_t i = 0; i < n; ++i) values(i, 0) = A(i, i);
    return values;
}

static void ml_sort_eig_descending(ms::ColMatrix<double>& values, ms::ColMatrix<double>& V) {
    const size_t n = values.rows();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return values(a, 0) > values(b, 0);
    });
    ms::ColMatrix<double> sv(n, 1, 0.0);
    ms::ColMatrix<double> sV(n, n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        sv(j, 0) = values(order[j], 0);
        for (size_t i = 0; i < n; ++i) sV(i, j) = V(i, order[j]);
    }
    values = std::move(sv);
    V = std::move(sV);
}

static Mat spectral_embedding_jacobi(const Mat& L, int k) {
    ms::ColMatrix<double> Lm = ml_mat_to_col(L);
    ms::ColMatrix<double> V;
    ms::ColMatrix<double> evals = ml_jacobi_eigen_symmetric(Lm, V);
    ml_sort_eig_descending(evals, V);
    int dim = (int)evals.rows();
    int n = (int)L.size();
    int use_k = std::min(k, dim);
    int start = std::max(0, dim - use_k);

    Mat emb(n, Vec(use_k, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < use_k; ++c)
            emb[i][c] = V((size_t)i, (size_t)(start + c));
        double rn = vec_norm(emb[i]);
        if (rn > 1e-14)
            for (int c = 0; c < use_k; ++c) emb[i][c] /= rn;
    }
    return emb;
}

static bool spectral_eig_sym_valid(const ms::EigResult& res) {
    if (res.values.rows() == 0) return false;
    double max_ev = std::abs(res.values(0, 0));
    return std::isfinite(max_ev) && max_ev > 1e-6 && max_ev <= 2.5;
}

static Mat spectral_embedding_from_eig(const ms::EigResult& res, int k) {
    const auto& evecs = res.vectors;
    int dim = (int)res.values.rows();
    int n = (int)evecs.rows();
    int use_k = std::min(k, dim);
    int start = std::max(0, dim - use_k);

    Mat emb(n, Vec(use_k, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < use_k; ++c)
            emb[i][c] = evecs((size_t)i, (size_t)(start + c));
        double rn = vec_norm(emb[i]);
        if (rn > 1e-14)
            for (int c = 0; c < use_k; ++c) emb[i][c] /= rn;
    }
    return emb;
}

std::vector<int> spectral_clustering(const Mat& X, int n_clusters,
                                       double sigma, int n_neighbors) {
    int n = (int)X.size();
    if (n == 0 || X[0].empty()) return {};

    int k = std::max(1, std::min(n_clusters, n));

    if (k == 1) return std::vector<int>((size_t)n, 0);

    Mat W = build_affinity(X, sigma, n_neighbors);
    Mat L = symmetric_normalized_laplacian(W);

    Mat embedding;
    auto eig_res = ms::eig_sym(ml_mat_to_col(L));
    bool used_eig_sym = eig_res.has_value() && spectral_eig_sym_valid(*eig_res);
    if (used_eig_sym)
        embedding = spectral_embedding_from_eig(*eig_res, k);
    else
        embedding = spectral_embedding_jacobi(L, k);

    KMeans km(k, 300, 1e-6);
    km.fit(embedding);
    Vec labels = km.predict(embedding);

    std::vector<int> out((size_t)n);
    for (int i = 0; i < n; ++i) out[(size_t)i] = (int)labels[(size_t)i];
    return out;
}

// ========================== Agglomerative Clustering ==========================

void AgglomerativeClustering::fit(const Mat& X) {
    int n=X.size();
    labels_.resize(n);
    // Start: each point is its own cluster
    std::vector<std::set<int>> clusters(n);
    for (int i=0;i<n;++i) clusters[i]={i};

    // Build distance matrix
    std::vector<std::vector<double>> D(n, std::vector<double>(n,0));
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) D[i][j]=D[j][i]=eucl_dist(X[i],X[j]);

    std::vector<int> active(n); std::iota(active.begin(),active.end(),0);

    while ((int)active.size()>n_clusters) {
        // Find closest pair
        double best=1e300; int bi=-1,bj=-1;
        for (size_t ai=0;ai<active.size();++ai) for (size_t aj=ai+1;aj<active.size();++aj) {
            int ci=active[ai],cj=active[aj];
            double d=0;
            if (linkage=="single") {
                d=1e300;
                for (int p:clusters[ci]) for (int q:clusters[cj]) d=std::min(d,D[p][q]);
            } else {  // average/ward
                for (int p:clusters[ci]) for (int q:clusters[cj]) d+=D[p][q];
                d/=clusters[ci].size()*clusters[cj].size();
            }
            if (d<best){best=d;bi=(int)ai;bj=(int)aj;}
        }
        // Merge bj into bi
        int ci=active[bi], cj=active[bj];
        clusters[ci].insert(clusters[cj].begin(),clusters[cj].end());
        active.erase(active.begin()+bj);
    }

    // Assign labels
    for (int li=0;li<(int)active.size();++li)
        for (int i:clusters[active[li]]) labels_[i]=li;
}

// ========================== PCA ==========================

static void svd_power(const Mat& A, Mat& U, Vec& s, Mat& Vt, int k, int max_iter=100) {
    int m=A.size(), n=A[0].size();
    U.assign(k, Vec(m)); s.assign(k,0); Vt.assign(k, Vec(n));
    // Thin SVD via randomised power iteration
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0,1);
    for (int r=0;r<k;++r) {
        Vec v(n); for (auto& x:v) x=nd(rng);
        // Normalize
        double nn=vec_norm(v); for (auto& x:v) x/=nn;
        // Power iteration: (AA^T)^p v = A (A^T A)^p-1 A^T v
        for (int it=0;it<max_iter;++it) {
            // u = A v
            Vec u(m,0);
            for (int i=0;i<m;++i) for (int j=0;j<n;++j) u[i]+=A[i][j]*v[j];
            // Deflate previous
            for (int prev=0;prev<r;++prev) {
                double d=vec_dot(U[prev],u);
                for (int i=0;i<m;++i) u[i]-=d*U[prev][i];
            }
            nn=vec_norm(u); if (nn<1e-14) break;
            for (auto& x:u) x/=nn;
            // v = A^T u
            Vec nv(n,0);
            for (int j=0;j<n;++j) for (int i=0;i<m;++i) nv[j]+=A[i][j]*u[i];
            for (int prev=0;prev<r;++prev) {
                double d=vec_dot(Vt[prev],nv);
                for (int j=0;j<n;++j) nv[j]-=d*Vt[prev][j];
            }
            nn=vec_norm(nv); if (nn<1e-14) break;
            for (auto& x:nv) x/=nn;
            v=nv;
        }
        // Compute sigma
        Vec u(m,0);
        for (int i=0;i<m;++i) for (int j=0;j<n;++j) u[i]+=A[i][j]*v[j];
        for (int prev=0;prev<r;++prev) {
            double d=vec_dot(U[prev],u);
            for (int i=0;i<m;++i) u[i]-=d*U[prev][i];
        }
        double sigma=vec_norm(u);
        if (sigma>1e-14) for (auto& x:u) x/=sigma;
        s[r]=sigma; U[r]=u; Vt[r]=v;
    }
}

void PCA::fit(const Mat& X) {
    int n=X.size(), p=X[0].size();
    // Center
    mean_.assign(p,0);
    for (auto& row:X) for (int j=0;j<p;++j) mean_[j]+=row[j];
    for (auto& m:mean_) m/=n;
    Mat Xc(n, Vec(p));
    for (int i=0;i<n;++i) for (int j=0;j<p;++j) Xc[i][j]=X[i][j]-mean_[j];
    // SVD of Xc / sqrt(n-1) → components = Vt rows
    int k=n_components;
    Mat U; Vec s; Mat Vt;
    svd_power(Xc, U, s, Vt, k);
    components=Vt;  // k × p
    explained_variance.assign(k,0);
    for (int r=0;r<k;++r) explained_variance[r]=s[r]*s[r]/(n-1);
}

Mat PCA::transform(const Mat& X) const {
    int n=X.size(), k=components.size();
    Mat Z(n, Vec(k,0));
    for (int i=0;i<n;++i) {
        Vec xc(X[i].size());
        for (size_t j=0;j<X[i].size();++j) xc[j]=X[i][j]-mean_[j];
        for (int r=0;r<k;++r) Z[i][r]=vec_dot(components[r],xc);
    }
    return Z;
}
Mat PCA::fit_transform(const Mat& X) { fit(X); return transform(X); }
Mat PCA::inverse_transform(const Mat& Z) const {
    int n=Z.size(), k=components.size(), p=mean_.size();
    Mat Xr(n, Vec(p,0));
    for (int i=0;i<n;++i) {
        for (int r=0;r<k;++r)
            for (int j=0;j<p;++j) Xr[i][j]+=Z[i][r]*components[r][j];
        for (int j=0;j<p;++j) Xr[i][j]+=mean_[j];
    }
    return Xr;
}

// ========================== t-SNE (simplified Barnes-Hut stub) ==========================

Mat TSNE::fit_transform(const Mat& X) const {
    int n=X.size();
    // Compute pairwise affinities
    std::vector<std::vector<double>> P(n, std::vector<double>(n,0));
    for (int i=0;i<n;++i) {
        double sigmai=1.0;
        // Binary search for sigma matching perplexity
        double lo=0.01,hi=100.0;
        for (int bs=0;bs<50;++bs) {
            double sig=(lo+hi)/2;
            double sum=0;
            for (int j=0;j<n;++j) if (j!=i) {
                double d2=0; for (size_t d=0;d<X[i].size();++d) d2+=(X[i][d]-X[j][d])*(X[i][d]-X[j][d]);
                sum+=std::exp(-d2/(2*sig*sig));
            }
            double H=0;
            for (int j=0;j<n;++j) if (j!=i) {
                double d2=0; for (size_t d=0;d<X[i].size();++d) d2+=(X[i][d]-X[j][d])*(X[i][d]-X[j][d]);
                double pij=std::exp(-d2/(2*sig*sig))/(sum+1e-12);
                if (pij>1e-12) H-=pij*std::log(pij);
            }
            if (H>std::log(perplexity)) lo=sig; else hi=sig;
            sigmai=sig;
        }
        double sum=0;
        for (int j=0;j<n;++j) if (j!=i) {
            double d2=0; for (size_t d=0;d<X[i].size();++d) d2+=(X[i][d]-X[j][d])*(X[i][d]-X[j][d]);
            P[i][j]=std::exp(-d2/(2*sigmai*sigmai)); sum+=P[i][j];
        }
        for (int j=0;j<n;++j) P[i][j]/=(sum+1e-12);
    }
    // Symmetrise
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
        P[i][j]=(P[i][j]+P[j][i])/(2*n);
    }
    // Initialise Y randomly
    std::mt19937 rng(42); std::normal_distribution<double> nd(0,0.0001);
    Mat Y(n, Vec(n_components));
    for (auto& row:Y) for (auto& v:row) v=nd(rng);
    Mat Y_prev=Y, gains(n, Vec(n_components,1.0));
    double mom=0.5;
    for (int iter=0;iter<max_iter;++iter) {
        if (iter==250) mom=0.8;
        // Compute Q
        double qsum=0;
        std::vector<std::vector<double>> Q(n,std::vector<double>(n,0));
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
            double d2=0; for (int d=0;d<n_components;++d) d2+=(Y[i][d]-Y[j][d])*(Y[i][d]-Y[j][d]);
            Q[i][j]=Q[j][i]=1.0/(1.0+d2); qsum+=2*Q[i][j];
        }
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) Q[i][j]/=(qsum+1e-12);
        // Gradient
        Mat dY(n, Vec(n_components,0));
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) if (j!=i) {
            double d2=0; for (int d=0;d<n_components;++d) d2+=(Y[i][d]-Y[j][d])*(Y[i][d]-Y[j][d]);
            double fac=4*(P[i][j]-Q[i][j])/(1.0+d2);
            for (int d=0;d<n_components;++d) dY[i][d]+=fac*(Y[i][d]-Y[j][d]);
        }
        // Update
        Mat Ynew(n, Vec(n_components));
        for (int i=0;i<n;++i) for (int d=0;d<n_components;++d)
            Ynew[i][d]=Y[i][d]-lr*dY[i][d]+mom*(Y[i][d]-Y_prev[i][d]);
        Y_prev=Y; Y=Ynew;
    }
    return Y;
}

// ========================== Autodiff ==========================

Var::Var(double v) : node(std::make_shared<ADNode>()) { node->val=v; }

// Topological sort of the computation graph rooted at n
static void build_topo(const std::shared_ptr<ADNode>& n,
                        std::vector<std::shared_ptr<ADNode>>& topo,
                        std::set<ADNode*>& visited) {
    if (visited.count(n.get())) return;
    visited.insert(n.get());
    for (auto& p : n->prev) build_topo(p, topo, visited);
    topo.push_back(n);
}

void Var::backward() {
    // Topological sort + reverse-mode backprop
    std::vector<std::shared_ptr<ADNode>> topo;
    std::set<ADNode*> visited;
    build_topo(node, topo, visited);
    node->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}

Var Var::operator+(const Var& o) const {
    Var out(val()+o.val());
    out.node->prev = {node, o.node};
    auto a=node, b=o.node, c=out.node;
    c->backward_fn=[a,b,c](){a->grad+=c->grad; b->grad+=c->grad;};
    return out;
}
Var Var::operator-(const Var& o) const {
    Var out(val()-o.val());
    out.node->prev = {node, o.node};
    auto a=node, b=o.node, c=out.node;
    c->backward_fn=[a,b,c](){a->grad+=c->grad; b->grad-=c->grad;};
    return out;
}
Var Var::operator*(const Var& o) const {
    Var out(val()*o.val());
    out.node->prev = {node, o.node};
    auto a=node, b=o.node, c=out.node;
    c->backward_fn=[a,b,c](){a->grad+=b->val*c->grad; b->grad+=a->val*c->grad;};
    return out;
}
Var Var::operator/(const Var& o) const {
    Var out(val()/o.val());
    out.node->prev = {node, o.node};
    auto a=node, b=o.node, c=out.node;
    double bv=o.val();
    c->backward_fn=[a,b,bv,c](){a->grad+=c->grad/bv; b->grad-=a->val*c->grad/(bv*bv);};
    return out;
}
Var Var::operator-() const {
    Var out(-val());
    out.node->prev = {node};
    auto a=node, c=out.node;
    c->backward_fn=[a,c](){a->grad-=c->grad;};
    return out;
}
Var var_exp(const Var& x) {
    double ev=std::exp(x.val());
    Var out(ev); out.node->prev={x.node};
    auto a=x.node, c=out.node;
    c->backward_fn=[a,c,ev](){a->grad+=ev*c->grad;};
    return out;
}
Var var_log(const Var& x) {
    Var out(std::log(x.val())); out.node->prev={x.node};
    auto a=x.node, c=out.node; double xv=x.val();
    c->backward_fn=[a,c,xv](){a->grad+=c->grad/xv;};
    return out;
}
Var var_tanh(const Var& x) {
    double tv=std::tanh(x.val());
    Var out(tv); out.node->prev={x.node};
    auto a=x.node, c=out.node; double d=1-tv*tv;
    c->backward_fn=[a,c,d](){a->grad+=d*c->grad;};
    return out;
}
Var var_sigmoid(const Var& x) {
    double sv=1.0/(1.0+std::exp(-x.val()));
    Var out(sv); out.node->prev={x.node};
    auto a=x.node, c=out.node; double d=sv*(1-sv);
    c->backward_fn=[a,c,d](){a->grad+=d*c->grad;};
    return out;
}
Var var_relu(const Var& x) {
    double xv=x.val();
    Var out(xv>0?xv:0.0); out.node->prev={x.node};
    auto a=x.node, c=out.node;
    c->backward_fn=[a,c,xv](){if(xv>0) a->grad+=c->grad;};
    return out;
}
Var var_sqrt(const Var& x) {
    double sv=std::sqrt(x.val());
    Var out(sv); out.node->prev={x.node};
    auto a=x.node, c=out.node;
    c->backward_fn=[a,c,sv](){a->grad+=c->grad/(2*sv+1e-12);};
    return out;
}
Var var_pow(const Var& x, double n) {
    double pv=std::pow(x.val(),n);
    Var out(pv); out.node->prev={x.node};
    auto a=x.node, c=out.node; double xv=x.val();
    c->backward_fn=[a,c,n,xv](){a->grad+=n*std::pow(xv,n-1)*c->grad;};
    return out;
}

// Numerical gradient helper for functions not using Var
Vec grad(std::function<Var(const std::vector<Var>&)> f, const Vec& x) {
    int n=x.size(); Vec g(n);
    for (int i=0;i<n;++i) {
        std::vector<Var> xv(n);
        for (int j=0;j<n;++j) { xv[j]=Var(x[j]); xv[j].node->grad=0; }
        auto y=f(xv);
        y.backward();
        g[i]=xv[i].node->grad;
    }
    return g;
}

Mat jacobian(std::function<std::vector<Var>(const std::vector<Var>&)> F, const Vec& x) {
    int n=x.size();
    // Forward evaluation to get output size
    std::vector<Var> xv(n); for (int i=0;i<n;++i) xv[i]=Var(x[i]);
    auto yv=F(xv); int m=yv.size();
    Mat J(m, Vec(n,0));
    for (int k=0;k<m;++k) {
        std::vector<Var> xv2(n); for (int i=0;i<n;++i) { xv2[i]=Var(x[i]); xv2[i].node->grad=0; }
        auto yv2=F(xv2);
        yv2[k].backward();
        for (int i=0;i<n;++i) J[k][i]=xv2[i].node->grad;
    }
    return J;
}

Mat hessian(std::function<Var(const std::vector<Var>&)> f, const Vec& x) {
    double h=1e-5; int n=x.size(); Mat H(n,Vec(n));
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
        Vec xpp=x,xpm=x,xmp=x,xmm=x;
        xpp[i]+=h;xpp[j]+=h; xpm[i]+=h;xpm[j]-=h;
        xmp[i]-=h;xmp[j]+=h; xmm[i]-=h;xmm[j]-=h;
        std::vector<Var> vpp,vpm,vmp,vmm;
        for (int k=0;k<n;++k){ vpp.push_back(Var(xpp[k]));vpm.push_back(Var(xpm[k]));vmp.push_back(Var(xmp[k]));vmm.push_back(Var(xmm[k])); }
        H[i][j]=(f(vpp).val()-f(vpm).val()-f(vmp).val()+f(vmm).val())/(4*h*h);
    }
    return H;
}

// ========================== Neural Network ==========================

static double apply_act(double z, const std::string& act) {
    if (act=="relu") return z>0?z:0;
    if (act=="sigmoid") return 1.0/(1.0+std::exp(-z));
    if (act=="tanh") return std::tanh(z);
    if (act=="softmax") return z;  // handled specially
    return z;  // linear
}
static double act_deriv(double z, const std::string& act) {
    if (act=="relu") return z>0?1.0:0.0;
    if (act=="sigmoid") { double s=1.0/(1.0+std::exp(-z)); return s*(1-s); }
    if (act=="tanh") { double t=std::tanh(z); return 1-t*t; }
    return 1.0;
}

DenseLayer::DenseLayer(int in_dim, int out_dim, std::string act)
    : activation(std::move(act)) {
    // He init
    std::mt19937 rng(42); std::normal_distribution<double> nd(0, std::sqrt(2.0/in_dim));
    W.assign(out_dim, Vec(in_dim)); b.assign(out_dim,0);
    for (auto& row:W) for (auto& v:row) v=nd(rng);
    // Adam state
    mW.assign(out_dim,Vec(in_dim,0)); vW.assign(out_dim,Vec(in_dim,0));
    mb.assign(out_dim,0); vb.assign(out_dim,0);
}

Vec DenseLayer::forward(const Vec& x) const {
    Vec z(W.size(),0);
    for (size_t i=0;i<W.size();++i) { z[i]=b[i]; for (size_t j=0;j<x.size();++j) z[i]+=W[i][j]*x[j]; }
    return z;
}
Vec DenseLayer::forward_activate(const Vec& z) const {
    Vec a(z.size());
    if (activation=="softmax") {
        double mx=*std::max_element(z.begin(),z.end()), sum=0;
        for (size_t i=0;i<z.size();++i) { a[i]=std::exp(z[i]-mx); sum+=a[i]; }
        for (auto& v:a) v/=sum;
    } else {
        for (size_t i=0;i<z.size();++i) a[i]=apply_act(z[i],activation);
    }
    return a;
}

NeuralNet& NeuralNet::add(int out_dim, std::string activation) {
    if (layers.empty()) {
        // defer — need input_dim from first batch
        layers.emplace_back(1, out_dim, std::move(activation));
    } else {
        int in_dim=layers.back().W[0].size();  // Wrong, fix at compile time
        (void)in_dim;
        layers.emplace_back(1, out_dim, std::move(activation));
    }
    return *this;
}

void NeuralNet::compile(std::string loss, double learning_rate) {
    loss_name=std::move(loss); lr=learning_rate;
}

void NeuralNet::fit(const Mat& X, const Mat& Y, int epochs, int batch) {
    int n=X.size(), in_dim=X[0].size(), out_dim=Y[0].size();
    // (Re-)init layers with correct dimensions
    std::vector<int> dims;
    dims.push_back(in_dim);
    std::vector<std::string> acts;
    for (auto& l:layers) { dims.push_back(l.W.size()); acts.push_back(l.activation); }
    dims.back()=out_dim;  // last layer output
    if (acts.empty()) { acts.push_back("relu"); dims.push_back(out_dim); }
    layers.clear();
    for (size_t li=0;li<acts.size();++li)
        layers.emplace_back(dims[li],dims[li+1],acts[li]);

    input_dim=in_dim;
    std::mt19937 rng(42);
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    for (int ep=0;ep<epochs;++ep) {
        std::shuffle(idx.begin(),idx.end(),rng);
        for (int start=0;start<n;start+=batch) {
            int end=std::min(start+batch,n);
            for (int i=start;i<end;++i) backward(X[idx[i]], Y[idx[i]]);
        }
    }
}

void NeuralNet::backward(const Vec& x, const Vec& target) {
    // Forward pass: store z and a for each layer
    int L=layers.size();
    std::vector<Vec> zs(L), as(L+1);
    as[0]=x;
    for (int l=0;l<L;++l) {
        zs[l]=layers[l].forward(as[l]);
        as[l+1]=layers[l].forward_activate(zs[l]);
    }
    // Compute loss gradient at output
    Vec delta(as[L].size());
    if (loss_name=="mse") {
        for (size_t i=0;i<as[L].size();++i) delta[i]=2.0*(as[L][i]-target[i])/as[L].size();
    } else {  // cross-entropy w/ softmax
        for (size_t i=0;i<as[L].size();++i) delta[i]=as[L][i]-target[i];
    }
    // Backprop
    ++t;
    double beta1=0.9, beta2=0.999, eps=1e-8;
    double lr_t=lr*std::sqrt(1-std::pow(beta2,t))/(1-std::pow(beta1,t));
    for (int l=L-1;l>=0;--l) {
        // delta * act'(z)
        if (layers[l].activation!="softmax") {
            for (size_t i=0;i<delta.size();++i) delta[i]*=act_deriv(zs[l][i],layers[l].activation);
        }
        // Gradient of W and b
        int out=layers[l].W.size(), in=layers[l].W[0].size();
        Vec grad_b=delta;
        Mat grad_W(out, Vec(in,0));
        for (int i=0;i<out;++i) for (int j=0;j<in;++j) grad_W[i][j]=delta[i]*as[l][j];
        // Adam update
        for (int i=0;i<out;++i) for (int j=0;j<in;++j) {
            layers[l].mW[i][j]=beta1*layers[l].mW[i][j]+(1-beta1)*grad_W[i][j];
            layers[l].vW[i][j]=beta2*layers[l].vW[i][j]+(1-beta2)*grad_W[i][j]*grad_W[i][j];
            layers[l].W[i][j]-=lr_t*layers[l].mW[i][j]/(std::sqrt(layers[l].vW[i][j])+eps);
        }
        for (int i=0;i<out;++i) {
            layers[l].mb[i]=beta1*layers[l].mb[i]+(1-beta1)*grad_b[i];
            layers[l].vb[i]=beta2*layers[l].vb[i]+(1-beta2)*grad_b[i]*grad_b[i];
            layers[l].b[i]-=lr_t*layers[l].mb[i]/(std::sqrt(layers[l].vb[i])+eps);
        }
        // Propagate delta back
        if (l>0) {
            Vec new_delta(in,0);
            for (int j=0;j<in;++j) for (int i=0;i<out;++i) new_delta[j]+=layers[l].W[i][j]*delta[i];
            delta=new_delta;
        }
    }
}

Mat NeuralNet::predict(const Mat& X) {
    Mat out; int L=layers.size();
    for (auto& x:X) {
        Vec a=x;
        for (int l=0;l<L;++l) a=layers[l].forward_activate(layers[l].forward(a));
        out.push_back(a);
    }
    return out;
}

// ========================== Loss Functions ==========================

double mse_loss(const Vec& yp, const Vec& yt) {
    double s=0; for (size_t i=0;i<yp.size();++i) s+=(yp[i]-yt[i])*(yp[i]-yt[i]);
    return s/yp.size();
}
double mae_loss(const Vec& yp, const Vec& yt) {
    double s=0; for (size_t i=0;i<yp.size();++i) s+=std::abs(yp[i]-yt[i]);
    return s/yp.size();
}
double binary_crossentropy(const Vec& yp, const Vec& yt) {
    double s=0; const double eps=1e-12;
    for (size_t i=0;i<yp.size();++i)
        s+=-(yt[i]*std::log(yp[i]+eps)+(1-yt[i])*std::log(1-yp[i]+eps));
    return s/yp.size();
}
double categorical_crossentropy(const Mat& yp, const Mat& yt) {
    double s=0; const double eps=1e-12;
    for (size_t i=0;i<yp.size();++i)
        for (size_t j=0;j<yp[i].size();++j) s+=-yt[i][j]*std::log(yp[i][j]+eps);
    return s/yp.size();
}
double huber_loss(const Vec& yp, const Vec& yt, double delta) {
    double s=0;
    for (size_t i=0;i<yp.size();++i) {
        double e=std::abs(yp[i]-yt[i]);
        s+=e<=delta ? 0.5*e*e : delta*(e-0.5*delta);
    }
    return s/yp.size();
}
double hinge_loss(const Vec& yp, const Vec& yt) {
    double s=0;
    for (size_t i=0;i<yp.size();++i) s+=std::max(0.0,1.0-yt[i]*yp[i]);
    return s/yp.size();
}

// ========================== Metrics ==========================

double accuracy(const Vec& yp, const Vec& yt) {
    int correct=0; for (size_t i=0;i<yp.size();++i) if (std::abs(yp[i]-yt[i])<0.5) ++correct;
    return (double)correct/yp.size();
}
double r2_score(const Vec& yp, const Vec& yt) {
    double mean_y=0; for (double v:yt) mean_y+=v; mean_y/=yt.size();
    double ss_res=0, ss_tot=0;
    for (size_t i=0;i<yt.size();++i){ ss_res+=(yt[i]-yp[i])*(yt[i]-yp[i]); ss_tot+=(yt[i]-mean_y)*(yt[i]-mean_y); }
    return 1-ss_res/(ss_tot+1e-12);
}
double rmse(const Vec& yp, const Vec& yt) { return std::sqrt(mse_loss(yp,yt)); }
double precision(const Vec& yp, const Vec& yt, double thr) {
    int tp=0,fp=0; for (size_t i=0;i<yp.size();++i) {
        if (yp[i]>=thr){ if (yt[i]>=thr) ++tp; else ++fp; }
    }
    return (double)tp/(tp+fp+1e-12);
}
double recall(const Vec& yp, const Vec& yt, double thr) {
    int tp=0,fn=0; for (size_t i=0;i<yp.size();++i) {
        if (yt[i]>=thr){ if (yp[i]>=thr) ++tp; else ++fn; }
    }
    return (double)tp/(tp+fn+1e-12);
}
double f1_score(const Vec& yp, const Vec& yt, double thr) {
    double p=precision(yp,yt,thr), r=recall(yp,yt,thr);
    return 2*p*r/(p+r+1e-12);
}

ConfusionMatrix confusion_matrix(const Vec& yp, const Vec& yt, double thr) {
    ConfusionMatrix cm;
    if (yp.size()!=yt.size()) return cm;
    for (size_t i=0;i<yp.size();++i) {
        bool pred_pos = yp[i]>=thr;
        bool true_pos = yt[i]>=0.5;
        if (pred_pos && true_pos) ++cm.tp;
        else if (pred_pos && !true_pos) ++cm.fp;
        else if (!pred_pos && !true_pos) ++cm.tn;
        else ++cm.fn;
    }
    return cm;
}

std::vector<ROCPoint> roc_curve(const Vec& yp, const Vec& yt) {
    std::vector<ROCPoint> pts;
    if (yp.size()!=yt.size() || yp.empty()) return pts;

    std::vector<double> cand(yp.begin(), yp.end());
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    // Trivial endpoints: a threshold above every score classifies nothing as
    // positive, one below every score classifies everything as positive.
    cand.push_back(cand.back()+1.0);
    cand.push_back(cand.front()-1.0);
    std::sort(cand.begin(), cand.end(), std::greater<double>());

    pts.reserve(cand.size());
    for (double thr : cand) {
        ConfusionMatrix cm = confusion_matrix(yp, yt, thr);
        double tpr = (cm.tp+cm.fn) > 0 ? (double)cm.tp/(cm.tp+cm.fn) : 0.0;
        double fpr = (cm.fp+cm.tn) > 0 ? (double)cm.fp/(cm.fp+cm.tn) : 0.0;
        pts.push_back({fpr, tpr, thr});
    }
    std::sort(pts.begin(), pts.end(), [](const ROCPoint& a, const ROCPoint& b){
        if (a.fpr!=b.fpr) return a.fpr<b.fpr;
        return a.tpr<b.tpr;
    });
    pts.erase(std::unique(pts.begin(), pts.end(), [](const ROCPoint& a, const ROCPoint& b){
        return a.fpr==b.fpr && a.tpr==b.tpr;
    }), pts.end());
    return pts;
}

double roc_auc(const Vec& yp, const Vec& yt) {
    auto pts = roc_curve(yp, yt);
    if (pts.size()<2) return 0.0;
    double auc=0.0;
    for (size_t i=1;i<pts.size();++i)
        auc += 0.5*(pts[i-1].tpr+pts[i].tpr)*(pts[i].fpr-pts[i-1].fpr);
    return auc;
}

std::vector<PRPoint> precision_recall_curve(const Vec& yp, const Vec& yt) {
    std::vector<PRPoint> pts;
    if (yp.size()!=yt.size() || yp.empty()) return pts;

    std::vector<double> cand(yp.begin(), yp.end());
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    cand.push_back(cand.back()+1.0);
    cand.push_back(cand.front()-1.0);
    std::sort(cand.begin(), cand.end(), std::greater<double>());

    pts.reserve(cand.size());
    for (double thr : cand) {
        ConfusionMatrix cm = confusion_matrix(yp, yt, thr);
        double rec = (cm.tp+cm.fn) > 0 ? (double)cm.tp/(cm.tp+cm.fn) : 0.0;
        double prec = (double)cm.tp/(cm.tp+cm.fp+1e-12);
        pts.push_back({prec, rec, thr});
    }
    std::sort(pts.begin(), pts.end(), [](const PRPoint& a, const PRPoint& b){
        if (a.recall!=b.recall) return a.recall<b.recall;
        return a.precision<b.precision;
    });
    pts.erase(std::unique(pts.begin(), pts.end(), [](const PRPoint& a, const PRPoint& b){
        return a.recall==b.recall && a.precision==b.precision;
    }), pts.end());
    // At each recall level keep the maximum precision (interpolated-PR /
    // scikit-learn average_precision_score convention).
    std::vector<PRPoint> collapsed;
    collapsed.reserve(pts.size());
    for (const auto& pt : pts) {
        if (!collapsed.empty() && collapsed.back().recall==pt.recall)
            collapsed.back().precision=std::max(collapsed.back().precision, pt.precision);
        else
            collapsed.push_back(pt);
    }
    return collapsed;
}

double average_precision(const Vec& yp, const Vec& yt) {
    auto pts = precision_recall_curve(yp, yt);
    if (pts.size()<2) return 0.0;
    double ap=0.0;
    for (size_t i=1;i<pts.size();++i)
        ap += (pts[i].recall-pts[i-1].recall)*pts[i].precision;
    return ap;
}

// ========================== Preprocessing ==========================

void StandardScaler::fit(const Mat& X) {
    int p=X[0].size(); mean_.assign(p,0); std_.assign(p,0);
    for (auto& row:X) for (int j=0;j<p;++j) mean_[j]+=row[j];
    for (auto& m:mean_) m/=X.size();
    for (auto& row:X) for (int j=0;j<p;++j) std_[j]+=(row[j]-mean_[j])*(row[j]-mean_[j]);
    for (auto& s:std_) s=std::sqrt(s/X.size()+1e-12);
}
Mat StandardScaler::transform(const Mat& X) const {
    int p=mean_.size(); Mat Z(X.size(),Vec(p));
    for (size_t i=0;i<X.size();++i) for (int j=0;j<p;++j) Z[i][j]=(X[i][j]-mean_[j])/std_[j];
    return Z;
}
Mat StandardScaler::fit_transform(const Mat& X) { fit(X); return transform(X); }
Mat StandardScaler::inverse_transform(const Mat& Z) const {
    int p=mean_.size(); Mat X(Z.size(),Vec(p));
    for (size_t i=0;i<Z.size();++i) for (int j=0;j<p;++j) X[i][j]=Z[i][j]*std_[j]+mean_[j];
    return X;
}
void MinMaxScaler::fit(const Mat& X) {
    int p=X[0].size();
    min_.assign(p,std::numeric_limits<double>::infinity());
    max_.assign(p,-std::numeric_limits<double>::infinity());
    for (auto& row:X) for (int j=0;j<p;++j){ min_[j]=std::min(min_[j],row[j]); max_[j]=std::max(max_[j],row[j]); }
}
Mat MinMaxScaler::transform(const Mat& X) const {
    int p=min_.size(); Mat Z(X.size(),Vec(p));
    for (size_t i=0;i<X.size();++i) for (int j=0;j<p;++j) Z[i][j]=(X[i][j]-min_[j])/(max_[j]-min_[j]+1e-12);
    return Z;
}
Mat MinMaxScaler::fit_transform(const Mat& X){ fit(X); return transform(X); }

// ========================== Cross-validation ==========================

std::pair<std::pair<Mat,Vec>,std::pair<Mat,Vec>>
train_test_split(const Mat& X, const Vec& y, double test_size, int seed) {
    int n=X.size();
    std::mt19937 rng(seed);
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),rng);
    int n_test=std::max(1,(int)(n*test_size));
    Mat Xtr,Xte; Vec ytr,yte;
    for (int i=0;i<n;++i){
        if (i<n_test){Xte.push_back(X[idx[i]]);yte.push_back(y[idx[i]]);}
        else {Xtr.push_back(X[idx[i]]);ytr.push_back(y[idx[i]]);}
    }
    return {{Xtr,ytr},{Xte,yte}};
}

Vec cross_val_score(std::function<double(const Mat&,const Vec&,const Mat&,const Vec&)> scorer,
                    const Mat& X, const Vec& y, int cv) {
    int n=X.size(), fold=n/cv;
    Vec scores;
    for (int k=0;k<cv;++k) {
        Mat Xte,Xtr; Vec yte,ytr;
        for (int i=0;i<n;++i) {
            if (i>=k*fold && i<(k+1)*fold){Xte.push_back(X[i]);yte.push_back(y[i]);}
            else {Xtr.push_back(X[i]);ytr.push_back(y[i]);}
        }
        if (!Xtr.empty()&&!Xte.empty()) scores.push_back(scorer(Xtr,ytr,Xte,yte));
    }
    return scores;
}

} // namespace ml
} // namespace ms
