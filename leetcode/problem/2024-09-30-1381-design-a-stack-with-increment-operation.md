---
layout: leetcode-entry
title: "1381. Design a Stack With Increment Operation"
permalink: "/leetcode/problem/2024-09-30-1381-design-a-stack-with-increment-operation/"
leetcode_ui: true
entry_slug: "2024-09-30-1381-design-a-stack-with-increment-operation"
---

[1381. Design a Stack With Increment Operation](https://leetcode.com/problems/design-a-stack-with-increment-operation/description/) medium
[blog post](https://leetcode.com/problems/design-a-stack-with-increment-operation/solutions/5850904/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30092024-1381-design-a-stack-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Y5hBYwmX8UU)
![1.webp](/assets/leetcode_daily_images/8e2db1f1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/751

#### Problem TLDR

Stack with range increment operation #medium #design

#### Intuition

The naive solution with a single array and O(n) increment operation is accepted.

The clever one is to maintain a second array for `increments` and compute them only for the `pop` operation, shrinking it by one position. Only the `last` increment matters for the stack top.

#### Approach

* let's implement both solutions

#### Complexity

- Time complexity:
$$O(n)$$ for n calls

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class CustomStack(maxSize: Int) {
    val arr = IntArray(maxSize); var head = 0
    fun push(x: Int) {
        if (head < arr.size) arr[head++] = x }
    fun pop() = if (head == 0) -1 else arr[--head]
    fun increment(k: Int, v: Int) {
        for (i in 0..<min(k, head)) arr[i] += v }
}

```
```kotlin

class CustomStack(maxSize: Int) {
    val arr = IntArray(maxSize); var size = 0
    val inc = IntArray(maxSize + 1)
    fun push(x: Int) { if (size < arr.size) arr[size++] = x }
    fun pop() = if (size < 1) -1 else inc[size] + arr[size - 1].also {
        inc[size - 1] += inc[size]; inc[size--] = 0
    }
    fun increment(k: Int, v: Int) {  inc[min(k, size)] += v }
}

```
```rust

struct CustomStack(Vec<i32>, Vec<i32>, usize);
impl CustomStack {
    fn new(maxSize: i32) -> Self {
        Self(vec![0; maxSize as usize], vec![0; maxSize as usize + 1], 0) }
    fn push(&mut self, x: i32) {
        if self.2 < self.0.len() { self.0[self.2] = x; self.2 += 1 } }
    fn pop(&mut self) -> i32 { if self.2 < 1 { -1 } else {
        let res = self.1[self.2] + self.0[self.2 -1];
        self.1[self.2 - 1] += self.1[self.2];
        self.1[self.2] = 0; self.2 -= 1;
        res }}
    fn increment(&mut self, k: i32, val: i32) {
        self.1[self.2.min(k as usize)] += val }
}

```
```c++

class CustomStack {
public:
    vector<int> arr, inc; int size;
    CustomStack(int maxSize): arr(maxSize), inc(maxSize + 1), size(0){}
    void push(int x) { if (size < arr.size()) arr[size++] = x; }
    int pop() {
        if (size < 1) return -1;
        int res = inc[size] + arr[size - 1];
        inc[size - 1] += inc[size]; inc[size--] = 0;
        return res;
    }
    void increment(int k, int val) { inc[min(k, size)] += val; }
};

```

