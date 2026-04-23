---
layout: leetcode-entry
title: "2411. Smallest Subarrays With Maximum Bitwise OR"
permalink: "/leetcode/problem/2025-07-29-2411-smallest-subarrays-with-maximum-bitwise-or/"
leetcode_ui: true
entry_slug: "2025-07-29-2411-smallest-subarrays-with-maximum-bitwise-or"
---

[2411. Smallest Subarrays With Maximum Bitwise OR](https://leetcode.com/problems/smallest-subarrays-with-maximum-bitwise-or/description/) medium
[blog post](https://leetcode.com/problems/smallest-subarrays-with-maximum-bitwise-or/solutions/7019034/kotlin-rust-by-samoylenkodmitry-fboa/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29072025-2411-smallest-subarrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/iyZBADt0P_U)
![1.webp](/assets/leetcode_daily_images/3b2f68a7.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1064

#### Problem TLDR

Shortest max OR i.. subarrays #medium #two_pointers

#### Intuition

Go backwards solutions:
1. Use bits frequency and two pointers: always expand, shrint while frequency is 2
2. Use nearest bit occurence map, then total length is the max occurence pointer

Go forwards solution:
* for each i go back and update answer[j--] while it is doing the update n[j] != n[j] | n[i]

#### Approach

* the bits frequency is the most template-like for two-pointers

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 72ms
    fun smallestSubarrays(n: IntArray): IntArray {
        val f = IntArray(32); val r = IntArray(n.size); var j = n.lastIndex
        for (i in n.lastIndex downTo 0) {
            for (b in 0..31) if (n[i] shr b and 1 > 0) ++f[b]
            while (i < j && (0..31).all { b -> f[b] > 1 || n[j] shr b and 1 < 1 }) {
                for (b in 0..31) if (n[j] shr b and 1 > 0) --f[b]
                j--
            }
            r[i] = j - i + 1
        }
        return r
    }

```
```kotlin

// 29ms
    fun smallestSubarrays(n: IntArray): IntArray {
        val j = IntArray(30); val r = IntArray(n.size)
        for (i in n.lastIndex downTo 0) {
            for (b in 0..29) if (n[i] shr b and 1 > 0) j[b] = i
            r[i] = max(1, j.max() - i + 1)
        }
        return r
    }

```
```kotlin

// 6ms
    fun smallestSubarrays(n: IntArray): IntArray {
        val r = IntArray(n.size) { 1 }
        for (i in n.indices) {
            var j = i - 1
            while (j >= 0 && n[j] != n[i] or n[j]) {
                n[j] = n[j] or n[i]
                r[j] = i - j-- + 1
            }
        }
        return r
    }

```
```rust

// 13ms
    pub fn smallest_subarrays(n: Vec<i32>) -> Vec<i32> {
        let (mut j, mut r) = ([0;30], vec![0; n.len()]);
        for i in (0..n.len()).rev() {
            for b in 0..30 { if n[i] >> b & 1 > 0 { j[b] = i } }
            r[i] = 1.max(1 + *j.iter().max().unwrap() as i32 - i as i32)
        } r
    }

```
```c++

// 11ms
    vector<int> smallestSubarrays(vector<int>& n) {
        int j[30]={}, m = 0, k; vector<int> r(size(n));
        for (int i = size(n) - 1; i >= 0; --i, m = 0) {
            for (int b = 0; b < 30; ++b) k = n[i] >> b & 1, m = max(m, j[b] = max(j[b] * (1 - k), i * k));
            r[i] = max(1, m - i + 1);
        } return r;
    }

```
```python3

// 588ms
    def smallestSubarrays(self, n: List[int]) -> List[int]:
        last = [0] * 30
        res = []
        for i in range(len(n) - 1, -1, -1):
            for b in range(30):
                if n[i] >> b & 1:
                    last[b] = i
            res.append(max(1, max(last) - i + 1))
        return res[::-1]

```
