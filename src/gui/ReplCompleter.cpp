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
        QStringLiteral("dist_solve"),
        QStringLiteral("dist_cg"),
        QStringLiteral("dist_gmres"),
        QStringLiteral("dist_jacobi"),
        QStringLiteral("dist_matmul"),
        QStringLiteral("transpose"),
        QStringLiteral("chol"),
        QStringLiteral("lu"),
        QStringLiteral("qr"),
        QStringLiteral("pinv"),
        QStringLiteral("null"),
        QStringLiteral("orth"),
        QStringLiteral("eig"),
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
        QStringLiteral("crypto_x25519_shared"),
        QStringLiteral("crypto_ed25519_keypair"),
        QStringLiteral("crypto_ed25519_sign"),
        QStringLiteral("crypto_ed25519_verify"),
        QStringLiteral("crypto_constant_time_eq"),
        QStringLiteral("crypto_random_bytes"),
        QStringLiteral("crypto_hkdf_sha256"),
        QStringLiteral("crypto_hkdf_sha512"),
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
        // ML / graph (common)
        QStringLiteral("ml_accuracy"),
        QStringLiteral("ml_rmse"),
        QStringLiteral("graph_pagerank"),
        QStringLiteral("graph_k_core_decomposition"),
        QStringLiteral("graph_k_core_subgraph"),
        QStringLiteral("graph_chromatic_number"),
        QStringLiteral("geo_minkowski_sum"),
        QStringLiteral("geo_clip_polygon"),
        QStringLiteral("geo_min_bounding_rect"),
        // Framework hooks
        QStringLiteral("izaac"),
        QStringLiteral("gria"),
        QStringLiteral("axiom"),
        // MPI helpers
        QStringLiteral("mpi_rank"),
        QStringLiteral("mpi_size"),
        QStringLiteral("mpi_allreduce_sum"),
        QStringLiteral("cuda_allreduce_sum"),
        QStringLiteral("cuda_allreduce_max"),
        QStringLiteral("cuda_allreduce_min"),
        QStringLiteral("cuda_allreduce_prod"),
        QStringLiteral("cuda_allreduce_avg"),
        QStringLiteral("cuda_broadcast"),
        QStringLiteral("cuda_reduce"),
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
