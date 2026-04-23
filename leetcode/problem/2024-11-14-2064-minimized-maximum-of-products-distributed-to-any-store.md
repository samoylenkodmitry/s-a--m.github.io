---
layout: leetcode-entry
title: "2064. Minimized Maximum of Products Distributed to Any Store"
permalink: "/leetcode/problem/2024-11-14-2064-minimized-maximum-of-products-distributed-to-any-store/"
leetcode_ui: true
entry_slug: "2024-11-14-2064-minimized-maximum-of-products-distributed-to-any-store"
---

[2064. Minimized Maximum of Products Distributed to Any Store](https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/description/) medium
[blog post](https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/solutions/6043709/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14112024-2064-minimized-maximum-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CFvh7vrvPU8)
[deep-dive](https://notebooklm.google.com/notebook/1d1e496d-c3fa-40be-a27d-e8eb28fb6da4/audio)
![1.webp](/assets/leetcode_daily_images/21fb34c3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/800

#### Problem TLDR

Min of max of `quantities` spread by `n` single-type stores #medium #binary_search #heap

#### Intuition

We can choose the `maximum` for each store and count how many `stores` are needed. The number of stores grows linearly with the increase of maximum, so we can do a Binary Search in a space of `max = 1..100_000`.

Another way of thinking: spread all `quantities` each on a single store: `q1 -> a, q2 -> b, q3 -> c, empty -> d`. Then choose peek the type with `maximum` single value in a store `1 + (quantity - 1)/stores_spread` and increas it's spread into one more store. This can be done with a PriorityQueue.

#### Approach

* let's do greedy-heap solution in Kotlin
* Binary Search in c++
* golf in Rust (it's time is still like a clean Binary Search though)

#### Complexity

- Time complexity:
$$O(mlog(M))$$ for Binary Search, O(nlog(m)) for Heap (slower)

- Space complexity:
$$O(1)$$ for Binary Search, O(n) for Heap

#### Code

```kotlin

    fun minimizedMaximum(n: Int, quantities: IntArray): Int {
        var l = 1; var h = 100000
        while (l <= h)
            if (n < quantities.sumBy { 1 + (it - 1) / ((l + h) / 2)})
            l = (l + h) / 2 + 1 else h = (l + h) / 2 - 1
        return l
    }

```
```kotlin

    fun minimizedMaximum(n: Int, quantities: IntArray): Int {
        val pq = PriorityQueue<IntArray>(compareBy { -(1 + (it[0] - 1) / it[1]) })
        for (i in 0..<n) pq +=
            if (i < quantities.size) intArrayOf(quantities[i], 1)
            else pq.poll().apply { this[1]++ }
        return 1 + (pq.peek()[0] - 1) / pq.peek()[1]
    }

```
```rust

    pub fn minimized_maximum(n: i32, quantities: Vec<i32>) -> i32 {
        Vec::from_iter(1..100001).partition_point(|x|
            n < quantities.iter().map(|&q| 1 + (q - 1) / x).sum()) as i32 + 1
    }

```
```c++

    int minimizedMaximum(int n, vector<int>& q) {
        int l = 1, r = 1e5, m, s;
        while (l <= r) {
            s = 0, m = (l + r) / 2;
            for (int x: q) s += (x + m - 1) / m;
            n < s ? l = m + 1 : r = m - 1;
        }
        return l;
    }

```

