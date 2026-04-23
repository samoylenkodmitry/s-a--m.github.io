---
layout: leetcode-entry
title: "2081. Sum of k-Mirror Numbers"
permalink: "/leetcode/problem/2025-06-23-2081-sum-of-k-mirror-numbers/"
leetcode_ui: true
entry_slug: "2025-06-23-2081-sum-of-k-mirror-numbers"
---

[2081. Sum of k-Mirror Numbers](https://leetcode.com/problems/sum-of-k-mirror-numbers/description) hard
[blog post](https://leetcode.com/problems/sum-of-k-mirror-numbers/solutions/6875610/kotlin-rust-by-samoylenkodmitry-81cg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23062025-2081-sum-of-k-mirror-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yW_A4NHIUYk)
![1.webp](/assets/leetcode_daily_images/fed0a4f7.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1028

#### Problem TLDR

n palindromes in both 10 and k-base #hard #math

#### Intuition

The simple intuition: brute-force all palindromes.
The trick is how to build palindromes in increasing order.

* subproblem: Closest Palindrome, hard (https://leetcode.com/problems/find-the-closest-palindrome/description/)

My own solution was accepted:
* iterate `halves` `1..some_max_value`
* build two possible tails, with doubled or not center value
* collect those values and then sort

The trick is how to find the `max_value` to be sure we got `all first n` values. I just brute-forced it and hardcoded, the function is `max(n) = f(2^x)` with some constants.

More optimal approach: generate palindromes in increasing order.
* iterate over `length` of the palindrome
* and iterate `halves = start..end`, where start is `10^half`, end is `10^(half+1)-1`
* then build a tail, and check

#### Approach

* even non optimal solution feels good if its your own
* however, let's try to understand and remember how to build the palindromes in order

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 1265ms
    fun kMirror(k: Int, n: Int): Long {
        val k = k.toLong(); val res = ArrayList<Long>()
        fun check(v: Long) {
            var x = v; var vk = 0L
            while (x > 0) { vk = vk * k + x % k; x /= k }
            if (vk == v) res += v
        }
        for (vd in 1..(1 shl ((7 * n + 166) / 18))) {
            val vd = vd.toLong(); var x = vd
            var v = vd; var v2 = vd
            while (x > 0) {
                v = v * 10L + x % 10L; x /= 10L
                if (x > 0) v2 = v2 * 10L + x % 10L
            }
            check(v); check(v2)
        }
        res.sort()
        return (0..<n).sumOf { res[it] }
    }

```
```kotlin

// 87ms
    fun kMirror(k: Int, n: Int): Long {
        var ans = 0L; var cnt = 0; var len = 0; val k = 1L * k
        while (cnt < n && ++len > 0) {
            val half = (len + 1) / 2
            var start = 1; for (i in 1..<half) start *= 10
            val end = start * 10 - 1;
            for (pref in start..end) {
                var pal = 1L * pref; var tail = 1L * pref
                if (len % 2 > 0) tail /= 10
                while (tail > 0) { pal = pal * 10 + (tail % 10); tail /= 10 }
                var t = pal; var rev = 0L
                while (t > 0) { rev = rev * k + (t % k); t /= k }
                if (rev == pal) { ans += pal; if (++cnt == n) break }
            }
        }
        return ans
    }

```

```rust

// 69ms
    pub fn k_mirror(k: i32, mut n: i32) -> i64 {
        let (mut ans, mut len, k) = (0i64, 0, k as i64);
        while n > 0 {
            len += 1; let half = (len + 1) / 2;
            let mut start = 1i64; for _ in 1..half { start *= 10; }
            let end = start * 10 - 1;
            for pref in start..=end {
                let mut pal = pref;
                let mut tail = if len % 2 == 0 { pref } else { pref / 10 };
                while tail > 0 { pal = pal * 10 + (tail % 10); tail /= 10; }
                let mut t = pal; let mut rev = 0i64;
                while t > 0 { rev = rev * k + (t % k); t /= k; }
                if rev == pal { ans += pal; n -= 1; if n == 0 { break } }
            }
        }  ans
    }

```

```c++

// 107ms
    long long kMirror(int k, int n) {
        long long ans = 0;
        for (int len = 1; n; ++len) {
            int halfLen = (len + 1) / 2;
            long long start = 1; for (int i = 1; i < halfLen; ++i) start *= 10;
            long long end = start * 10 - 1;
            for (long long prefix = start; prefix <= end && n; ++prefix) {
                long long pal = prefix; long long tail = prefix;
                if (len & 1) tail /= 10;
                while (tail) pal = pal * 10 + (tail % 10), tail /= 10;
                long long t = pal, rev = 0;
                while (t) rev = rev * k + (t % k), t /= k;
                if (rev == pal) ans += pal, --n;
            }
        }
        return ans;
    }

```

