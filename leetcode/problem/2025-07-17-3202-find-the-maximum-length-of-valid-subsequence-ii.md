---
layout: leetcode-entry
title: "3202. Find the Maximum Length of Valid Subsequence II"
permalink: "/leetcode/problem/2025-07-17-3202-find-the-maximum-length-of-valid-subsequence-ii/"
leetcode_ui: true
entry_slug: "2025-07-17-3202-find-the-maximum-length-of-valid-subsequence-ii"
---

[3202. Find the Maximum Length of Valid Subsequence II](https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-ii/description) medium
[blog post](https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-ii/solutions/6969808/kotlin-rust-by-samoylenkodmitry-yful/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17072025-3202-find-the-maximum-length?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FFoPyXxObBs)
![1.webp](/assets/leetcode_daily_images/d3ff2f53.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1052

#### Problem TLDR

Longest same-pair-k-parity subsequence #medium #dp

#### Intuition

Didn't solved.

What went right:
* arithmetics: `(a+b)%k == (b + c)%k, a%k == c%k`

What went wrong:
* spent some brainpower on dfs implementations (frist 15 minutes)
* the way to update length: closes i got is `++len[v][other]`, should've been `max(len[v][x%k], 1 + len[v][other])` (wasn't able to comprehend how to peek max and update current simulteneously)

Mostly irrelevant chain-of-thoughts:
```j
    // abcd
    // (a+b)%k == (b+c)%k
    // (a%k + b%k)%k == (b%k + c%k)%k
    // 1 4 1 4      %3    always ababab pattern, or aaaa ?
    //             (1+4)%3=2
    //                4%3=1, 1%3=1
    //             meet 4, look for (0..k) - 4%k

    // (a + b) % k = c
    // a % k = (c - b%k + k) % k

    // 1 4 2 3 1 4    k=3
    // *
    //   *           4: 1-4  start sequence parity  (1+4)%k=5%3=2
    //         *     for p=2: 2 - 1%k = 1

    // 1 2 3 4 5     k=2
    //     *

```

#### Approach

* no difference between iterations: `x in n v in 0..<k` or `v in 0..<k x in n`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

// 92ms
    fun maximumLength(n: IntArray, k: Int): Int {
        val len = Array(k) { IntArray(k) }
        for (v in 0..<k) for (x in n)
            len[v][x % k] = max(len[v][x % k], 1 + len[v][(k + v - (x % k)) % k])
        return len.maxOf { it.max() }
    }

```
```kotlin

// 92ms
    fun maximumLength(n: IntArray, k: Int): Int {
        val len = Array(k) { IntArray(k) }
        for (x in n) for (v in 0..<k)
            len[v][x % k] = max(len[v][x % k], 1 + len[v][(k + v - (x % k)) % k])
        return len.maxOf { it.max() }
    }

```

```rust

// 67ms
    pub fn maximum_length(n: Vec<i32>, k: i32) -> i32 {
        (0..k as usize).map(|v| {
            let k = k as usize; let mut len = vec![0; k];
            n.iter().map(|&x| { let x = x as usize;
                len[x % k] = len[x % k].max(1 + len[(k + v - x % k) % k]); len[x % k]
            }).max().unwrap() }).max().unwrap() as _
    }

```
```c++

// 55ms
    int maximumLength(vector<int>& n, int k) {
        int r = 0;
        for (int v = 0; v < k; ++v) for (int l[1000]={}; int x: n)
            r = max(r, l[x%k] = max(l[x%k], 1 + l[(v - x%k + k) % k]));
        return r;
    }

```

