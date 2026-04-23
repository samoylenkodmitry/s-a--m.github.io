---
layout: leetcode-entry
title: "1498. Number of Subsequences That Satisfy the Given Sum Condition"
permalink: "/leetcode/problem/2025-06-29-1498-number-of-subsequences-that-satisfy-the-given-sum-condition/"
leetcode_ui: true
entry_slug: "2025-06-29-1498-number-of-subsequences-that-satisfy-the-given-sum-condition"
---

[1498. Number of Subsequences That Satisfy the Given Sum Condition](https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/description/) medium
[blog post](https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/solutions/6898516/kotlin-rust-by-samoylenkodmitry-ba5s/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29062025-1498-number-of-subsequences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RZfJbcSmn6c)
![1.webp](/assets/leetcode_daily_images/16eacd4a.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1034

#### Problem TLDR

Subsequencies target in 0..min+max #medium #binary_search #two_pointers

#### Intuition

Observe the problem:

```j
    // 2,3,3,4,6,7 target = 10
    //       * * *

    // 7 2 7  72 27 2 727     -77 -7 -7
    // 2 7 7  27 27 2 277     -77 -7 -7
    // order doesn't matter, can sort

    // binary search target / 2 right border

    // for each value x find with bs t=(target - x)

    // 0 1 2 3 4 5
    // 2 3 3 4 6 7     t=12
    // j       i      6+2 =8 6+6=12
    // j         i    7+2 =9 7+7=14, 7+6=13, 7+4=11
    // f     t                               t=4
    // 4 + 3 + 2 + 1 = 10 = 4 * 5 / 2
```

* order doesn't matter -> can sort (subsequencies are different, but count is the same; for every min,max pair we can take any subset of others)
* now min..max is a subarray (not subsequence)
* at every position we count subarrays `ending` on that position
* count `good` or `bad`
* naive: binary search `target - n[i]`, bad is `i - j`
* clever: the right border only goes from the right to the left, no need for binary search

#### Approach

* optimize memory to O(1) using `2^x%m` exponentiation technique: `a^x = (a^2^x/2 + a^x%2)`
* write a short joke solution with `BigInteger`
* precompute `2^x` in one go counting `bad` subarrays

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 177ms
    fun numSubseq(n: IntArray, t: Int): Int {
        var i = 0; var j = n.size - 1; n.sort(); var c = 0.toBigInteger(); val one = 1.toBigInteger()
        while (i <= j) if (n[i] + n[j] > t) j-- else c = c.add(one.shiftLeft(j - i++))
        return c.mod(1_000_000_007.toBigInteger()).intValueExact()
    }

```
```kotlin

// 75ms
    fun numSubseq(n: IntArray, t: Int): Int {
        val M = 1_000_000_007; n.sort(); var cnt = 0
        val f = IntArray(n.size + 2); f[1] = 1
        for ((i, x) in n.withIndex()) {
            f[i + 2] = (f[i + 1] * 2) % M
            var lo = 0; var hi = i; var j = -1
            while (lo <= hi) {
                val m = (lo + hi) / 2
                if (x + n[m] <= t) { j = max(j, m); lo = m + 1 }
                else hi = m - 1
            }
            cnt = (cnt + f[i - j]) % M
        }
        return (f[n.size + 1] - cnt - 1 + M) % M
    }

```
```kotlin

// 57ms
    fun numSubseq(n: IntArray, t: Int): Int {
        val M = 1_000_000_007; n.sort(); var cnt = 0
        val f = IntArray(n.size + 2); f[1] = 1; var j = n.size - 1
        for ((i, x) in n.withIndex()) {
            f[i + 2] = (f[i + 1] * 2) % M
            while (j >= 0 && x + n[j] > t) j--
            if (j <= i) cnt = (cnt + f[i - j]) % M
        }
        return (f[n.size + 1] - cnt - 1 + M) % M
    }

```
```kotlin

// 55ms
    fun numSubseq(n: IntArray, t: Int): Int {
        val M = 1_000_000_007; var c = 0; var i = 0; var j = n.size - 1; n.sort()
        fun f(a: Long, x: Int): Long = if (a == 2L && x < 63) (1L shl x) % M else
            if (x == 0) 1L else (f((a * a) % M, x / 2) * if (x % 2 > 0) a else 1) % M
        while (i <= j) if (n[i] + n[j] > t) j--
                       else c = (c + f(2L, j - i++).toInt()) % M
        return c
    }

```
```kotlin

// 47ms
    fun numSubseq(n: IntArray, t: Int): Int {
        val M = 1_000_000_007; val f = IntArray(n.size); f[0] = 1
        var c = 0; var i = 0; var j = n.size - 1; n.sort()
        for (i in 1..<n.size) f[i] = (2 * f[i - 1]) % M
        while (i <= j) if (n[i] + n[j] > t) j--
                       else c = (c + f[j - i++]) % M
        return c
    }

```
```kotlin

// 45ms
    fun numSubseq(n: IntArray, t: Int): Int {
        val M = 1_000_000_007; n.sort(); var cnt = 0
        val f = IntArray(n.size); f[0] = 1; var j = n.size - 1
        for (i in 1..<n.size) f[i] = (2 * f[i - 1]) % M
        for ((i, x) in n.withIndex()) {
            while (j >= 0 && x + n[j] > t) j--
            if (j < i) break
            cnt = (cnt + f[j - i]) % M
        }
        return cnt
    }

```
```rust

// 5ms
    pub fn num_subseq(mut n: Vec<i32>, t: i32) -> i32 {
        let (M, mut c, mut j) = (1_000_000_007, 0, n.len());
        n.sort_unstable(); let mut f = vec![0; n.len() + 2]; f[1] = 1;
        for (i, x) in n.iter().enumerate() {
            f[i + 2] = (f[i + 1] * 2) % M;
            while j > 0 && x + n[j - 1] > t { j -= 1 }
            if i + 1 >= j { c = (c + f[i + 1 - j]) % M }
        } (f[n.len() + 1] - c - 1 + M) % M
    }

```
```c++

// 0ms
    int numSubseq(vector<int>& a, int t) {
        int n = a.size(), m = 1e9+7; vector<int> p(n,1);
        sort(a.begin(), a.end());
        for(int i = 1; i < n; ++i)  p[i] = (p[i-1]*2) % m;
        int i = 0, j = n-1, r = 0;
        while(i <= j) if(a[i] + a[j] > t) --j;
                      else r = (r + p[j-i++]) % m;
        return r;
    }

```

