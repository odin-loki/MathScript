#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
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

static double eucl_dist(const Vec& a, const Vec& b) {
    double s=0; for (size_t i=0;i<a.size();++i) s+=(a[i]-b[i])*(a[i]-b[i]);
    return std::sqrt(s);
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
    for (int i=0;i<n;++i) {
        int ci=std::find(classes.begin(),classes.end(),y[i])-classes.begin();
        counts[ci]++;
        for (int j=0;j<p;++j) mean[ci][j]+=X[i][j];
    }
    for (int c=0;c<C;++c) { class_prior[c]=(double)counts[c]/n; for (int j=0;j<p;++j) mean[c][j]/=counts[c]; }
    for (int i=0;i<n;++i) {
        int ci=std::find(classes.begin(),classes.end(),y[i])-classes.begin();
        for (int j=0;j<p;++j) var[ci][j]+=(X[i][j]-mean[ci][j])*(X[i][j]-mean[ci][j]);
    }
    for (int c=0;c<C;++c) for (int j=0;j<p;++j) var[c][j]=var[c][j]/counts[c]+1e-9;
}
Vec NaiveBayes::predict(const Mat& X) const {
    Vec pred(X.size());
    int C=classes.size(), p=X[0].size();
    for (size_t i=0;i<X.size();++i) {
        double best=-1e300; int best_c=0;
        for (int c=0;c<C;++c) {
            double log_prob=std::log(class_prior[c]);
            for (int j=0;j<p;++j) {
                double diff=X[i][j]-mean[c][j];
                log_prob-=0.5*std::log(2*M_PI*var[c][j])+0.5*diff*diff/var[c][j];
            }
            if (log_prob>best){best=log_prob;best_c=c;}
        }
        pred[i]=classes[best_c];
    }
    return pred;
}
double NaiveBayes::score(const Mat& X, const Vec& y) const { return accuracy(predict(X),y); }

// ========================== Decision Tree ==========================

static double gini(const Vec& y, const std::vector<int>& idx) {
    if (idx.empty()) return 0;
    std::map<double,int> cnt;
    for (int i:idx) cnt[y[i]]++;
    double g=1.0;
    for (auto&[k,v]:cnt) { double p=(double)v/idx.size(); g-=p*p; }
    return g;
}

int DecisionTree::build(const Mat& X, const Vec& y, std::vector<int>& idx, int depth) {
    int node_idx=nodes.size();
    nodes.push_back({});
    if (depth>=max_depth || idx.size()<=1) {
        // Leaf: majority class
        std::map<double,int> cnt;
        for (int i:idx) cnt[y[i]]++;
        nodes[node_idx].value=std::max_element(cnt.begin(),cnt.end(),[](auto&a,auto&b){return a.second<b.second;})->first;
        return node_idx;
    }
    // Check if all same label
    bool same=true; for (int i:idx) if (y[i]!=y[idx[0]]) {same=false;break;}
    if (same) { nodes[node_idx].value=y[idx[0]]; return node_idx; }

    double best_g=1e300; int best_f=-1; double best_t=0;
    int p=X[0].size();
    for (int f=0;f<p;++f) {
        std::vector<double> vals; for (int i:idx) vals.push_back(X[i][f]);
        std::sort(vals.begin(),vals.end());
        vals.erase(std::unique(vals.begin(),vals.end()),vals.end());
        for (size_t vi=0;vi+1<vals.size();++vi) {
            double t=(vals[vi]+vals[vi+1])/2;
            std::vector<int> li,ri;
            for (int i:idx) (X[i][f]<=t?li:ri).push_back(i);
            if (li.empty()||ri.empty()) continue;
            double g=(li.size()*gini(y,li)+ri.size()*gini(y,ri))/idx.size();
            if (g<best_g){best_g=g;best_f=f;best_t=t;}
        }
    }
    if (best_f<0) {
        std::map<double,int> cnt; for (int i:idx) cnt[y[i]]++;
        nodes[node_idx].value=std::max_element(cnt.begin(),cnt.end(),[](auto&a,auto&b){return a.second<b.second;})->first;
        return node_idx;
    }
    nodes[node_idx].feature=best_f; nodes[node_idx].threshold=best_t;
    std::vector<int> li,ri;
    for (int i:idx) (X[i][best_f]<=best_t?li:ri).push_back(i);
    int li_node=nodes.size(); // will be set after left build
    nodes[node_idx].left=build(X,y,li,depth+1);
    nodes[node_idx].right=build(X,y,ri,depth+1);
    (void)li_node;
    return node_idx;
}

void DecisionTree::fit(const Mat& X, const Vec& y) {
    nodes.clear();
    std::vector<int> idx(X.size()); std::iota(idx.begin(),idx.end(),0);
    build(X,y,idx,0);
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
double DecisionTree::score(const Mat& X, const Vec& y) const { return accuracy(predict(X),y); }

// ========================== KMeans ==========================

void KMeans::fit(const Mat& X) {
    int n=X.size(), p=X[0].size();
    // K-means++ init
    std::mt19937 rng(42);
    centers.clear();
    std::uniform_int_distribution<int> uid(0,n-1);
    centers.push_back(X[uid(rng)]);
    for (int ci=1;ci<k;++ci) {
        Vec dists(n);
        for (int i=0;i<n;++i) {
            double best=1e300;
            for (auto&c:centers) best=std::min(best,eucl_dist(X[i],c)*eucl_dist(X[i],c));
            dists[i]=best;
        }
        double tot=0; for (double d:dists) tot+=d;
        std::uniform_real_distribution<double> ud(0,tot);
        double r=ud(rng); double cum=0;
        for (int i=0;i<n;++i) { cum+=dists[i]; if (cum>=r){centers.push_back(X[i]);break;} }
        if ((int)centers.size()<ci+1) centers.push_back(X[n-1]);
    }

    labels_.resize(n);
    for (int iter=0;iter<max_iter;++iter) {
        // Assign
        bool changed=false;
        for (int i=0;i<n;++i) {
            double best=1e300; int bi=0;
            for (int c=0;c<k;++c) { double d=eucl_dist(X[i],centers[c]); if (d<best){best=d;bi=c;} }
            if (labels_[i]!=bi) changed=true;
            labels_[i]=bi;
        }
        // Update
        Mat new_centers(k, Vec(p,0));
        std::vector<int> cnt(k,0);
        for (int i=0;i<n;++i) { cnt[labels_[i]]++; for (int j=0;j<p;++j) new_centers[labels_[i]][j]+=X[i][j]; }
        for (int c=0;c<k;++c) if (cnt[c]>0) for (int j=0;j<p;++j) new_centers[c][j]/=cnt[c];
        centers=new_centers;
        if (!changed) break;
    }
}
Vec KMeans::predict(const Mat& X) const {
    Vec pred(X.size());
    for (size_t i=0;i<X.size();++i) {
        double best=1e300; int bi=0;
        for (int c=0;c<k;++c) { double d=eucl_dist(X[i],centers[c]); if (d<best){best=d;bi=c;} }
        pred[i]=bi;
    }
    return pred;
}
double KMeans::inertia(const Mat& X) const {
    double s=0;
    for (size_t i=0;i<X.size();++i) { double d=eucl_dist(X[i],centers[(int)labels_[i]]); s+=d*d; }
    return s;
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
