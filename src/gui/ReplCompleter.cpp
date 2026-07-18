#include "gui/ReplCompleter.hpp"

#include <QAbstractItemView>
#include <QCompleter>
#include <QLineEdit>
#include <QShortcut>
#include <QStringListModel>

namespace gui {
namespace {

bool is_command_char(QChar ch) {
    return ch.isLetterOrNumber() || ch == '_';
}

int word_start(const QString& text, int cursor_pos) {
    int start = cursor_pos;
    while (start > 0 && is_command_char(text[start - 1])) {
        --start;
    }
    return start;
}

QString word_at_cursor(const QLineEdit* input) {
    const QString text = input->text();
    const int pos = input->cursorPosition();
    const int start = word_start(text, pos);
    return text.mid(start, pos - start);
}

class ReplCommandCompleter : public QCompleter {
public:
    explicit ReplCommandCompleter(QLineEdit* input, QStringList commands)
        : QCompleter(new QStringListModel(std::move(commands), input), input), input_(input) {
        setCaseSensitivity(Qt::CaseInsensitive);
        setCompletionMode(QCompleter::PopupCompletion);
        setFilterMode(Qt::MatchContains);
        setMaxVisibleItems(12);
        if (popup() != nullptr) {
            popup()->setAlternatingRowColors(true);
        }
    }

    QString pathFromIndex(const QModelIndex& index) const override {
        const QString selected = model()->data(index, Qt::DisplayRole).toString();
        const QString text = input_->text();
        const int pos = input_->cursorPosition();
        const int start = word_start(text, pos);
        return text.left(start) + selected + text.mid(pos);
    }

    QStringList splitPath(const QString& /*path*/) const override {
        return {word_at_cursor(input_)};
    }

private:
    QLineEdit* input_ = nullptr;
};

QStringList repl_command_list() {
    return {
        // REPL meta commands
        QStringLiteral("help"),
        QStringLiteral("vars"),
        QStringLiteral("version"),
        QStringLiteral("show"),
        QStringLiteral("saveplot"),
        QStringLiteral("clear"),
        QStringLiteral("topology"),
        QStringLiteral("simd"),
        QStringLiteral("dispatch"),
        QStringLiteral("balance"),
        QStringLiteral("gpu"),
        QStringLiteral("mpi"),
        QStringLiteral("frameworks"),
        QStringLiteral("exit"),
        QStringLiteral("save"),
        QStringLiteral("load"),
        QStringLiteral("run_file"),
        QStringLiteral("source"),
        // Linear algebra
        QStringLiteral("matmul"),
        QStringLiteral("solve"),
        QStringLiteral("bicgstab"),
        QStringLiteral("qmr"),
        QStringLiteral("lsqr"),
        QStringLiteral("tfqmr"),
        QStringLiteral("lsmr"),
        QStringLiteral("minres"),
        QStringLiteral("solve_sylvester"),
        QStringLiteral("dist_solve"),
        QStringLiteral("dist_cg"),
        QStringLiteral("dist_gmres"),
        QStringLiteral("dist_jacobi"),
        QStringLiteral("dist_matmul"),
        QStringLiteral("transpose"),
        QStringLiteral("chol"),
        QStringLiteral("expm"),
        QStringLiteral("sqrtm"),
        QStringLiteral("logm"),
        QStringLiteral("tril"),
        QStringLiteral("triu"),
        QStringLiteral("cosm"),
        QStringLiteral("sinm"),
        QStringLiteral("lu"),
        QStringLiteral("qr"),
        QStringLiteral("hess"),
        QStringLiteral("schur"),
        QStringLiteral("bidiag"),
        QStringLiteral("pinv"),
        QStringLiteral("null"),
        QStringLiteral("orth"),
        QStringLiteral("eig"),
        QStringLiteral("eig_sym"),
        QStringLiteral("ldl"),
        QStringLiteral("svd"),
        QStringLiteral("kron"),
        QStringLiteral("repmat"),
        QStringLiteral("linspace"),
        QStringLiteral("zeros"),
        QStringLiteral("ones"),
        QStringLiteral("rand"),
        // Plotting
        QStringLiteral("plot"),
        QStringLiteral("plot3d"),
        // Symbolic
        QStringLiteral("sym_simplify"),
        QStringLiteral("sym_expand"),
        QStringLiteral("sym_diff"),
        QStringLiteral("sym_integrate"),
        QStringLiteral("sym_eval"),
        QStringLiteral("sym_collect"),
        QStringLiteral("sym_solve_linear"),
        QStringLiteral("sym_substitute"),
        QStringLiteral("sym_limit"),
        QStringLiteral("sym_series"),
        QStringLiteral("sym_laplace"),
        QStringLiteral("sym_ilaplace"),
        QStringLiteral("sym_fourier"),
        QStringLiteral("sym_ifourier"),
        QStringLiteral("sym_dsolve"),
        // Crypto
        QStringLiteral("crypto_aes128_encrypt_block"),
        QStringLiteral("crypto_aes128_decrypt_block"),
        QStringLiteral("crypto_aes256_encrypt_block"),
        QStringLiteral("crypto_aes256_decrypt_block"),
        QStringLiteral("crypto_aes128_cbc_encrypt"),
        QStringLiteral("crypto_aes128_cbc_decrypt"),
        QStringLiteral("crypto_aes256_cbc_encrypt"),
        QStringLiteral("crypto_aes256_cbc_decrypt"),
        QStringLiteral("crypto_aes128_gcm_encrypt"),
        QStringLiteral("crypto_aes128_gcm_decrypt"),
        QStringLiteral("crypto_aes256_gcm_encrypt"),
        QStringLiteral("crypto_aes256_gcm_decrypt"),
        QStringLiteral("crypto_chacha20"),
        QStringLiteral("crypto_chacha20_poly1305_encrypt"),
        QStringLiteral("crypto_chacha20_poly1305_decrypt"),
        QStringLiteral("crypto_x25519_keypair"),
        QStringLiteral("crypto_x25519_shared"),
        QStringLiteral("crypto_ed25519_keypair"),
        QStringLiteral("crypto_ed25519_sign"),
        QStringLiteral("crypto_ed25519_verify"),
        QStringLiteral("crypto_constant_time_eq"),
        QStringLiteral("crypto_random_bytes"),
        QStringLiteral("crypto_hkdf_sha256"),
        QStringLiteral("crypto_hkdf_sha512"),
        QStringLiteral("crypto_sha256"),
        QStringLiteral("crypto_sha512"),
        QStringLiteral("crypto_hmac_sha256"),
        QStringLiteral("crypto_hmac_sha512"),
        QStringLiteral("crypto_pbkdf2_sha256"),
        QStringLiteral("crypto_pbkdf2_hmac_sha512"),
        // FFT / signal
        QStringLiteral("fft"),
        QStringLiteral("ifft"),
        QStringLiteral("fft2"),
        QStringLiteral("ifft2"),
        QStringLiteral("signal_unwrap"),
        QStringLiteral("signal_upsample"),
        QStringLiteral("signal_downsample"),
        QStringLiteral("signal_decimate"),
        QStringLiteral("signal_interpolate"),
        QStringLiteral("signal_resample"),
        QStringLiteral("signal_savgol"),
        QStringLiteral("signal_median_filter"),
        QStringLiteral("signal_coherence"),
        QStringLiteral("signal_lms"),
        QStringLiteral("signal_lms_weights"),
        QStringLiteral("signal_filter"),
        QStringLiteral("signal_filtfilt"),
        QStringLiteral("signal_sosfilt"),
        QStringLiteral("signal_conv2"),
        QStringLiteral("signal_deconv"),
        QStringLiteral("signal_czt"),
        QStringLiteral("signal_czt_zoom"),
        QStringLiteral("signal_firwin"),
        QStringLiteral("signal_firwin_highpass"),
        // Stats
        QStringLiteral("stats_linear_regression"),
        QStringLiteral("stats_pacf"),
        QStringLiteral("stats_arfit"),
        QStringLiteral("stats_multiple_regression"),
        QStringLiteral("stats_vif"),
        QStringLiteral("stats_variance_inflation_factor"),
        QStringLiteral("stats_partial_correlation"),
        QStringLiteral("stats_weighted_mean"),
        QStringLiteral("stats_trimmed_mean"),
        QStringLiteral("stats_kde"),
        QStringLiteral("stats_bootstrap_ci"),
        // Info theory (Wave 258)
        QStringLiteral("info_permutation_entropy"),
        QStringLiteral("info_transfer_entropy"),
        // Probability (extended Wave 257)
        QStringLiteral("prob_lognormal_pdf"),
        QStringLiteral("prob_lognormal_cdf"),
        QStringLiteral("prob_lognormal_ppf"),
        QStringLiteral("prob_weibull_pdf"),
        QStringLiteral("prob_weibull_cdf"),
        QStringLiteral("prob_weibull_ppf"),
        QStringLiteral("prob_laplace_pdf"),
        QStringLiteral("prob_laplace_cdf"),
        QStringLiteral("prob_laplace_ppf"),
        QStringLiteral("prob_logistic_pdf"),
        QStringLiteral("prob_logistic_cdf"),
        QStringLiteral("prob_logistic_ppf"),
        QStringLiteral("prob_gumbel_pdf"),
        QStringLiteral("prob_gumbel_cdf"),
        QStringLiteral("prob_gumbel_ppf"),
        QStringLiteral("prob_cauchy_pdf"),
        QStringLiteral("prob_cauchy_cdf"),
        QStringLiteral("prob_cauchy_ppf"),
        QStringLiteral("prob_pareto_pdf"),
        QStringLiteral("prob_pareto_cdf"),
        QStringLiteral("prob_pareto_ppf"),
        QStringLiteral("prob_rayleigh_pdf"),
        QStringLiteral("prob_rayleigh_cdf"),
        QStringLiteral("prob_rayleigh_ppf"),
        QStringLiteral("prob_gamma_cdf"),
        QStringLiteral("prob_beta_cdf"),
        QStringLiteral("prob_f_cdf"),
        // ML / graph (common)
        QStringLiteral("ml_accuracy"),
        QStringLiteral("ml_rmse"),
        QStringLiteral("numthy_factor"),
        QStringLiteral("graph_pagerank"),
        QStringLiteral("graph_dijkstra"),
        QStringLiteral("graph_bellman_ford"),
        QStringLiteral("graph_katz_centrality"),
        QStringLiteral("graph_algebraic_connectivity"),
        QStringLiteral("graph_adjacency_spectrum"),
        QStringLiteral("graph_laplacian"),
        QStringLiteral("graph_normalised_laplacian"),
        QStringLiteral("graph_modularity"),
        QStringLiteral("graph_eccentricity"),
        QStringLiteral("graph_is_strongly_connected"),
        QStringLiteral("graph_k_core_decomposition"),
        QStringLiteral("graph_k_core_subgraph"),
        QStringLiteral("graph_chromatic_number"),
        QStringLiteral("geo_minkowski_sum"),
        QStringLiteral("geo_clip_polygon"),
        QStringLiteral("geo_convex_hull"),
        QStringLiteral("geo_triangulate_polygon"),
        QStringLiteral("geo_convex_hull_3d"),
        QStringLiteral("geo_min_bounding_rect"),
        QStringLiteral("geo_kdtree_nearest"),
        QStringLiteral("geo_kdtree_3d_nearest"),
        QStringLiteral("geo_kdtree_knn"),
        QStringLiteral("geo_kdtree_range"),
        QStringLiteral("geo_point_in_aabb"),
        QStringLiteral("geo_overlap_aabb"),
        QStringLiteral("geo_intersect_seg_seg"),
        QStringLiteral("geo_intersect_ray_sphere"),
        QStringLiteral("geo_intersect_ray_aabb"),
        QStringLiteral("geo_intersect_ray_tri"),
        QStringLiteral("geo_dist_point_plane"),
        QStringLiteral("geo_dist_point_seg3d"),
        // Image transforms
        QStringLiteral("gray2rgb"),
        QStringLiteral("impad"),
        QStringLiteral("radon"),
        QStringLiteral("iradon"),
        QStringLiteral("imflip"),
        QStringLiteral("imrotate90"),
        QStringLiteral("threshold_binary"),
        QStringLiteral("adapthisteq"),
        QStringLiteral("imtophat"),
        QStringLiteral("imbothat"),
        QStringLiteral("imgradient_morph"),
        QStringLiteral("imadjust"),
        QStringLiteral("imhist"),
        QStringLiteral("label_components"),
        QStringLiteral("watershed"),
        QStringLiteral("slic"),
        QStringLiteral("hough_lines"),
        QStringLiteral("hough_circles"),
        QStringLiteral("harris"),
        QStringLiteral("shi_tomasi"),
        // Framework hooks
        QStringLiteral("izaac"),
        QStringLiteral("gria"),
        QStringLiteral("axiom"),
        // MPI helpers
        QStringLiteral("mpi_rank"),
        QStringLiteral("mpi_size"),
        QStringLiteral("mpi_allreduce_sum"),
        QStringLiteral("mpi_allreduce_max"),
        QStringLiteral("mpi_allreduce_min"),
        QStringLiteral("mpi_bcast"),
        QStringLiteral("mpi_barrier"),
        QStringLiteral("cuda_allreduce_sum"),
        QStringLiteral("cuda_allreduce_max"),
        QStringLiteral("cuda_allreduce_min"),
        QStringLiteral("cuda_allreduce_prod"),
        QStringLiteral("cuda_allreduce_avg"),
        QStringLiteral("cuda_broadcast"),
        QStringLiteral("cuda_reduce"),
        QStringLiteral("cuda_allgather"),
        QStringLiteral("cuda_nccl_available"),
        QStringLiteral("cuda_nccl_comm_size"),
        QStringLiteral("cuda_nccl_device_count"),
    };
}

}  // namespace

void setup_repl_completer(QLineEdit* input) {
    if (input == nullptr) {
        return;
    }

    auto* completer = new ReplCommandCompleter(input, repl_command_list());
    input->setCompleter(completer);

    const auto refresh_prefix = [input]() {
        if (auto* c = input->completer()) {
            c->setCompletionPrefix(word_at_cursor(input));
        }
    };
    QObject::connect(input, &QLineEdit::textChanged, input, refresh_prefix);
    QObject::connect(input, &QLineEdit::cursorPositionChanged, input, refresh_prefix);

    auto* force_complete = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Space), input);
    force_complete->setContext(Qt::WidgetShortcut);
    QObject::connect(force_complete, &QShortcut::activated, input, [input]() {
        if (auto* c = input->completer()) {
            c->setCompletionPrefix(word_at_cursor(input));
            c->complete();
        }
    });
}

}  // namespace gui
