#pragma once

#include <cstring>
#include <vector>

template <class T> class Ringbuffer {
public:
  /**
   * Ringbuffer constructor
   * @param number of elements
   **/
  explicit Ringbuffer(std::size_t maxElements)
      : data(maxElements), size(maxElements), head(0), tail(0), full(false) {}

  /**
   * Copy constructor
   **/
  Ringbuffer(const Ringbuffer &other) = delete; // default;

  /**
   * Move constructor
   **/
  Ringbuffer(Ringbuffer &&other) = default;

  /**
   * Assignment operator
   **/
  Ringbuffer &operator=(const Ringbuffer &other) = delete; // default;

  /**
   * Move assignment
   **/
  Ringbuffer &operator=(Ringbuffer &&other) = default;

  /**
   * Resets the ring buffer.
   */
  auto reset() -> void {
    head = 0;
    tail = 0;
    full = false;
  }

  /**
   * Returns max number of elements
   * @return max number of elements
   **/
  auto getSize() const -> std::size_t { return size; }

  /**
   * @return current number of elements
   **/
  auto getNumberOfElements() const -> std::size_t {
    return (full ? size : ((tail - head + size) % size));
  }

  /**
   * @return boolean indicating if there are any elements contained
   **/
  auto isEmpty() const -> bool { return ((head == tail) && !full); }

  /**
   * @return boolean indicating if the buffer is full
   **/
  auto isFull() const -> bool { return full; }

  /**
   * Add in one new elements
   * @param the element
   * @return boolean indicating success
   **/
  auto push(T element) -> bool {
    if (full)
      return false;
    data[tail] = element;
    tail = (tail + 1) % size;
    if (head == tail)
      full = true;
    return true;
  }

  /**
   * Remove one element
   * @param Pointer to element
   * @return boolean indicating success
   **/
  auto pop(T &element) -> bool {
    if (head == tail && !full)
      return false;
    full = false;
    element = data[head];
    head = (head + 1) % size;
    return true;
  }

private:
  std::vector<T> data;
  std::size_t size;
  std::size_t head;
  std::size_t tail;
  bool full;
};
