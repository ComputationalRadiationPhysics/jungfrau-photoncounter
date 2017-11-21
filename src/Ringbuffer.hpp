#pragma once

#include <cstring>
#include <memory>
#include <mutex>


template <class T> class RingBuffer {
public:
    RingBuffer(std::size_t maxElements)
        : size(maxElements), full(false), data(new T[size]), head(0), tail(0)
    {
    }

    RingBuffer(const RingBuffer& other)
        : size(other.size), full(other.full), data(new T[size]),
          head(other.head), tail(other.tail)
    {
        memcpy(other.data, data, size * sizeof(T));
    }

    RingBuffer& operator=(const RingBuffer& other) = delete;

    auto getSize() const -> std::size_t { return size; }

    auto getNumberOfElements() const -> std::size_t
    {
        return (full ? size : ((tail - head + size) % size));
    }

    auto isEmpty() const -> bool { return ((head == tail) && !full); }

    auto isFull() const -> bool { return full; }

    auto push(T element) -> bool
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (full)
            return false;
        data[tail] = element;
        tail = (++tail) % size;
        if (head == tail)
            full = true;
        return true;
    }

    auto pop(T& element) -> bool
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (head == tail && !full)
            return false;
        full = false;
        element = data[head];
        head = (++head) % size;
        return true;
    }

private:
    std::unique_ptr<T[]> data;
    std::size_t size;
    std::size_t head;
    std::size_t tail;
    std::mutex mutex;
    bool full;
};