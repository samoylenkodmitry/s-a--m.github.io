---
layout: leetcode-entry
title: "641. Design Circular Deque"
permalink: "/leetcode/problem/2024-09-28-641-design-circular-deque/"
leetcode_ui: true
entry_slug: "2024-09-28-641-design-circular-deque"
---

[641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/description/) medium
[blog post](https://leetcode.com/problems/design-circular-deque/solutions/5843028/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28092024-641-design-circular-deque?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JfjotYX4hBw)
![1.webp](/assets/leetcode_daily_images/7e3b563e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/749

#### Problem TLDR

Ring buffer #medium

#### Intuition

We can use a `Node` LinkedList-like data structure or a simple array with two pointers.

#### Approach

* `size` variable makes code simpler to reason about but can be omitted

#### Complexity

- Time complexity:
$$O(n)$$ for `n` calls to methods

- Space complexity:
$$O(k)$$

#### Code

```kotlin

class MyCircularDeque(k: Int) {
    var arr = IntArray(k + 1); var f = 1; var l = 0
    fun insertFront(value: Int) = !isFull().also { if (!it)
        { f = (arr.size + f - 1) % arr.size; arr[f] = value }}
    fun insertLast(value: Int) = !isFull().also { if (!it)
        { l = (l + 1) % arr.size; arr[l] = value }}
    fun deleteFront() = !isEmpty().also { if (!it) f = (f + 1) % arr.size }
    fun deleteLast() = !isEmpty().also { if (!it) l = (arr.size + l - 1) % arr.size }
    fun getFront() = if (isEmpty()) -1 else arr[f]
    fun getRear() = if (isEmpty()) -1 else arr[l]
    fun isEmpty() = size == 0
    fun isFull() = size == arr.size - 1
    val size get() = (arr.size + l - f + 1) % arr.size
}

```
```rust

struct MyCircularDeque((Vec<i32>, usize, usize));
impl MyCircularDeque {
    fn new(k: i32) -> Self { Self((vec![0; k as usize + 1], 1, 0)) }
    fn insert_front(&mut self, value: i32) -> bool { !self.is_full() && {
        self.0.1 = (self.0.0.len() + self.0.1 - 1) % self.0.0.len(); self.0.0[self.0.1] = value; true }}
    fn insert_last(&mut self, value: i32) -> bool { !self.is_full() && {
        self.0.2 = (self.0.2 + 1) % self.0.0.len(); self.0.0[self.0.2] = value ; true }}
    fn delete_front(&mut self) -> bool { !self.is_empty() && {
        self.0.1 = (self.0.1 + 1) % self.0.0.len(); true }}
    fn delete_last(&mut self) -> bool { !self.is_empty() && {
        self.0.2 = (self.0.0.len() + self.0.2 - 1) % self.0.0.len(); true }}
    fn get_front(&self) -> i32 { if self.is_empty() { -1 } else { self.0.0[self.0.1] }}
    fn get_rear(&self) -> i32 { if self.is_empty() { -1 } else { self.0.0[self.0.2] }}
    fn is_empty(&self) -> bool { self.size() == 0 }
    fn is_full(&self) -> bool { self.size() == self.0.0.len() - 1 }
    fn size(&self) -> usize { (self.0.0.len() + self.0.2 - self.0.1 + 1) % self.0.0.len() }
}

```
```c++

class MyCircularDeque {
public:
    vector<int> arr; int l, f;
    MyCircularDeque(int k) : arr(k + 1), l(0), f(1) {}
    bool insertFront(int value) { if (isFull()) return false;
        f = (arr.size() + f - 1) % arr.size(); arr[f] = value; return true; }
    bool insertLast(int value) { if (isFull()) return false;
        l = (l + 1) % arr.size(); arr[l] = value; return true; }
    bool deleteFront() { if (isEmpty()) return false;
        f = (f + 1) % arr.size(); return true; }
    bool deleteLast() { if (isEmpty()) return false;
        l = (arr.size() + l - 1) % arr.size(); return true; }
    int getFront() { return isEmpty() ? -1 : arr[f]; }
    int getRear() { return isEmpty() ? -1 : arr[l]; }
    bool isEmpty() { return size() == 0; }
    bool isFull() { return size() == arr.size() - 1; }
    int size() { return (arr.size() + l - f + 1) % arr.size(); }
};

```

