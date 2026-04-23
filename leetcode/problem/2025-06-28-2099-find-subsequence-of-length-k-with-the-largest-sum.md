---
layout: leetcode-entry
title: "2099. Find Subsequence of Length K With the Largest Sum"
permalink: "/leetcode/problem/2025-06-28-2099-find-subsequence-of-length-k-with-the-largest-sum/"
leetcode_ui: true
entry_slug: "2025-06-28-2099-find-subsequence-of-length-k-with-the-largest-sum"
---

[2099. Find Subsequence of Length K With the Largest Sum](https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/description/) easy
[blog post](https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/solutions/6894315/kotlin-rust-by-samoylenkodmitry-qegu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28062025-2099-find-subsequence-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/J_RbgxzZRA8)
![1.webp](/assets/leetcode_daily_images/33930fd2.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1033

#### Problem TLDR

Subsequence of k largest #easy #sort

#### Intuition

Sort, take k, restore original order.

#### Approach

* use sort or heap
* try to write quickselect (Hoare is the fastest)
* corner case is the duplicate numbers, count how many included in largest k

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 36ms
    fun maxSubsequence(n: IntArray, k: Int) = n
    .withIndex().sortedBy { -it.value }.take(k)
    .sortedBy { it.index }.map { it.value }

```
```kotlin

// 31ms
    fun maxSubsequence(n: IntArray, k: Int) =
        n.toMutableList().apply {
            while (size > k) remove(min())
        }

```
```kotlin

// 26ms
    fun maxSubsequence(n: IntArray, k: Int): List<Int> {
        val q = PriorityQueue<Pair<Int, Int>>(compareBy { it.first })
        for ((i, x) in n.withIndex()) {
            q += x to i
            if (q.size > k) q.poll()
        }
        return q.sortedBy { it.second }.map { it.first }
    }

```
```kotlin

// 17ms
    fun maxSubsequence(n: IntArray, k: Int): IntArray {
        val src = n.clone(); var i = 0
        var lo = 0; var hi = n.lastIndex
        while (lo < hi) {
            var l = lo; var h = hi
            val t = (n[lo] + n[hi]) / 2
            while (l <= h) {
                while (n[l] < t) ++l
                while (n[h] > t) --h
                if (l <= h) n[l] = n[h].also { n[h--] = n[l++] }
            }
            if (n.size - k > h) lo = l else hi = h
        }
        val min = n[n.size - k]
        var cnt = (n.size - k..<n.size).count { n[it] == min }
        return IntArray(k) { while (src[i] < min || src[i] == min && --cnt < 0) ++i; src[i++] }
    }

```
```rust

// 0ms
    pub fn max_subsequence(mut n: Vec<i32>, k: i32) -> Vec<i32> {
        for i in 0..n.len() { n[i] = (n[i] << 11) | i as i32 }
        n.sort_unstable(); let l = n.len() - k as usize;
        (&mut n[l..]).sort_unstable_by_key(|x| x & ((1 << 11) - 1));
        for x in &mut n { *x >>= 11 } n[l..].to_vec()
    }

```
```c++

// 0ms
    vector<int> maxSubsequence(vector<int>& a, int k) {
        auto b = a; sort(begin(b), end(b), greater<>());
        int m = b[k-1], c = count(begin(b), begin(b) + k, m);
        vector<int> r;
        for (int x: a) if (x > m || (x == m && c-- > 0)) r.push_back(x);
        return r;
    }

```

