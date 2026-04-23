---
layout: leetcode-entry
title: "873. Length of Longest Fibonacci Subsequence"
permalink: "/leetcode/problem/2025-02-27-873-length-of-longest-fibonacci-subsequence/"
leetcode_ui: true
entry_slug: "2025-02-27-873-length-of-longest-fibonacci-subsequence"
---

[873. Length of Longest Fibonacci Subsequence](https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/description/) medium
[blog post](https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/solutions/6473444/kotlin-rust-by-samoylenkodmitry-toq5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27022025-873-length-of-longest-fibonacci?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jxMF7xflM5c)
![1.webp](/assets/leetcode_daily_images/94773a26.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/908

#### Problem TLDR

Longest sequence a, b, a + b #medium #dynamic_programming

#### Intuition

Observing an example:

```j

    // 1,2,3,4,5,6,7,8
    // a b c
    //   a b   c
    //     a   b     c

```
The sequence length is always the same for any given (a, b), so can be cached.

#### Approach

* we can use set and check if next/previus is there
* we can use binary search making it O(1) for memory (c++ solution)
* the DP with HashMap of two numbers is slower than the Binary Search

#### Complexity

- Time complexity:
$$O(n^2)$$ or n^2log^2(n) for BinarySearch, n^2log(n) for HashSet

- Space complexity:
$$O(n)$$, O(1) for BinarySearch

#### Code

```kotlin

    fun lenLongestFibSubseq(arr: IntArray): Int {
        val s = arr.toSet(); var res = 0
        for (i in arr.indices) for (j in i + 1..<arr.size) {
            var a = arr[i]; var b = arr[j]; var l = 2
            while (a + b in s) { a = b.also { b = a + b }; l++ }
            res = max(res, l)
        }
        return if (res > 2) res else 0
    }

```
```rust

    pub fn len_longest_fib_subseq(arr: Vec<i32>) -> i32 {
        let (mut res, mut dp) = (0, HashMap::new());
        for i in 0..arr.len() { for j in i + 1..arr.len() {
            let b = arr[i]; let c = arr[j]; let a = c - b;
            let l = 1 + dp.get(&(a, b)).unwrap_or(&1);
            dp.insert((b, c), l); res = res.max(l)
        }}; if res > 2 { res } else { 0 }
    }

```
```c++

    int lenLongestFibSubseq(vector<int>& A) {
        int res = 0;
        for (int i = 0; i < size(A); i++) for (int j = i + 1; j < size(A); j++) {
            int a = A[i], b = A[j], l = 2;
            while (binary_search(begin(A), end(A), a + b)) tie(a, b, l) = tuple(b, a + b, l + 1);
            res = max(res, l);
        }
        return res > 2 ? res : 0;
    }

```

