#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class MageDpMatrix
{
public:
    static constexpr std::size_t FILE_BACKED_THRESHOLD = 50000;

    MageDpMatrix() = default;
    MageDpMatrix(std::size_t n, const std::string &filename)
    {
        reset(n, filename);
    }

    MageDpMatrix(const MageDpMatrix &) = delete;
    MageDpMatrix &operator=(const MageDpMatrix &) = delete;

    MageDpMatrix(MageDpMatrix &&other) noexcept
    {
        moveFrom(other);
    }

    MageDpMatrix &operator=(MageDpMatrix &&other) noexcept
    {
        if (this != &other)
        {
            closeStorage();
            moveFrom(other);
        }
        return *this;
    }

    ~MageDpMatrix()
    {
        closeStorage();
    }

    void reset(std::size_t n, const std::string &filename)
    {
        closeStorage();
        n_ = n;
        const std::size_t bytes = checkedBytes(n);

        if (n_ > FILE_BACKED_THRESHOLD)
        {
            if (filename.empty())
                throw std::invalid_argument("MAGE file-backed DP storage requires a non-empty filename");
            openFileBacked(filename, bytes);
        }
        else
        {
            memory_.assign(bytes, 0);
            data_ = memory_.data();
        }
    }

    std::size_t size() const { return n_; }

    uint8_t *row(std::size_t r)
    {
        return data_ + r * n_;
    }

    const uint8_t *row(std::size_t r) const
    {
        return data_ + r * n_;
    }

    uint8_t &at(std::size_t r, std::size_t c)
    {
        return data_[r * n_ + c];
    }

    const uint8_t &at(std::size_t r, std::size_t c) const
    {
        return data_[r * n_ + c];
    }

private:
    std::size_t n_ = 0;
    std::size_t bytes_ = 0;
    int fd_ = -1;
    uint8_t *data_ = nullptr;
    std::vector<uint8_t> memory_;

    static std::size_t checkedBytes(std::size_t n)
    {
        if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n)
            throw std::overflow_error("MAGE DP matrix size overflows size_t");
        return n * n;
    }

    static std::string errnoMessage(const std::string &action, const std::string &filename)
    {
        return action + " '" + filename + "': " + std::strerror(errno);
    }

    void openFileBacked(const std::string &filename, std::size_t bytes)
    {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
            throw std::overflow_error("MAGE DP matrix is too large for file-backed storage on this platform");

        fd_ = ::open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd_ == -1)
            throw std::runtime_error(errnoMessage("failed to open MAGE DP backing file", filename));

        bytes_ = bytes;
        if (::ftruncate(fd_, static_cast<off_t>(bytes_)) == -1)
        {
            const std::string message = errnoMessage("failed to size MAGE DP backing file", filename);
            closeStorage();
            throw std::runtime_error(message);
        }

        void *mapped = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED)
        {
            const std::string message = errnoMessage("failed to mmap MAGE DP backing file", filename);
            closeStorage();
            throw std::runtime_error(message);
        }
        data_ = static_cast<uint8_t *>(mapped);
    }

    void closeStorage()
    {
        if (data_ != nullptr && memory_.empty())
        {
            ::munmap(data_, bytes_);
        }
        if (fd_ != -1)
        {
            ::close(fd_);
        }
        memory_.clear();
        data_ = nullptr;
        fd_ = -1;
        bytes_ = 0;
        n_ = 0;
    }

    void moveFrom(MageDpMatrix &other)
    {
        n_ = other.n_;
        bytes_ = other.bytes_;
        fd_ = other.fd_;
        data_ = other.data_;
        memory_ = std::move(other.memory_);
        if (!memory_.empty())
            data_ = memory_.data();

        other.n_ = 0;
        other.bytes_ = 0;
        other.fd_ = -1;
        other.data_ = nullptr;
    }
};
