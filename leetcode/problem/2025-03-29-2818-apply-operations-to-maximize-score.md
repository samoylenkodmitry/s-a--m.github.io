---
layout: leetcode-entry
title: "2818. Apply Operations to Maximize Score"
permalink: "/leetcode/problem/2025-03-29-2818-apply-operations-to-maximize-score/"
leetcode_ui: true
entry_slug: "2025-03-29-2818-apply-operations-to-maximize-score"
---

[2818. Apply Operations to Maximize Score](https://leetcode.com/problems/apply-operations-to-maximize-score/description) hard
[blog post](https://leetcode.com/problems/apply-operations-to-maximize-score/solutions/6592302/kotlin-rust-by-samoylenkodmitry-kms3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29032025-2818-apply-operations-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Mt4Ge_AWoz8)
![1.webp](/assets/leetcode_daily_images/7f8db226.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/942

#### Problem TLDR

Multiply `k` numbers with max prime factor count #hard #monotonic_stack

#### Intuition

Didn't solve without the hints.
Take maximum values first.
We can take all subarrays where this value has the maximum prime factors count (and is leftmost).
`Count subarrays = count to the left * count to the right`
Use the monotonic stack to track left and right values: always add to the stack, remove all elements less than the current.

To build prime numbers: maintain boolean array up to 100_000, for every number that is prime mark all its multiplications as not prime.

To multiply `res = current ^ range % mod` use exponentiation technique: `x^e = x^(2*e/2) = (x^2)^(e/2) = (x * x) ^ (e / 2) = x^(e/2)*x^(e/2)` if e % 2 == 0, or x * x^(e/2) * x^(e/2) if not.

#### Approach

* we can inline the exponent function
* we can skip computing prime numbers, as O(nsqrt(n)) accepted

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or O(nsqrt(n))

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maximumScore(nums: List<Int>, k: Int): Int {
        val m = 1_000_000_007L; var res = 1L; var k = 1L * k
        val scores = nums.map { var x = it; var c = 0; var p = 2;
            while (p * p < x && x > 0) { if (x % p < 1) { c++; while (x % p == 0) x /= p }; p++ }
            c + if (x > 1) 1 else 0 }
        val left = IntArray(nums.size) { -1 }; val right = IntArray(nums.size) { nums.size }
        val l = ArrayList<Int>(); val r = ArrayList<Int>()
        for (i in nums.indices) { val j = nums.size - 1 - i
            while (l.size > 0 && scores[l.last()] < scores[i]) l.removeLast()
            while (r.size > 0 && scores[r.last()] <= scores[j]) r.removeLast()
            if (l.size > 0) left[i] = l.last(); if (r.size > 0) right[j] = r.last()
            l += i; r += j }
        for (i in nums.indices.sortedBy{ -nums[it] }) {
            val range = 1L * (i - left[i]) * (right[i] - i)
            var x = 1L * nums[i]; var e = 1L * min(range, k)
            while (e > 0) { if (e % 2 > 0) res = (res * x) % m; e /= 2; x = (x * x) % m }
            k -= range; if (k < 1L) break }
        return res.toInt()
    }

```
```rust

    pub fn maximum_score(nums: Vec<i32>, mut k: i32) -> i32 {
        let scores: Vec<_> = nums.iter().map(|&n| { let mut x = n; let mut c = 0; let mut p = 2;
            while p * p < x && x > 0 { if x % p < 1 { c += 1; while x % p == 0 { x /= p }}; p += 1 }
            c + ((x > 1) as i32) }).collect();
        let (n, m, mut res) = (nums.len(), 1_000_000_007, 1);
        let (mut left, mut right, mut l, mut r) = (vec![-1; n], vec![n as i64; n], vec![], vec![]);
        for i in 0..n { let j = n - 1 - i;
            while l.len() > 0 && scores[l[l.len() - 1]] < scores[i] { l.pop(); }
            while r.len() > 0 && scores[r[r.len() - 1]] <= scores[j] { r.pop(); }
            if l.len() > 0 { left[i] = l[l.len() - 1] as i64 }; if r.len() > 0 { right[j] = r[r.len() - 1] as i64 }
            l.push(i); r.push(j) }
        let mut idx: Vec<_> = (0..n).collect(); idx.sort_unstable_by_key(|&i| -nums[i]);
        for i in idx { let range = (i as i64 - left[i]) * (right[i] - i as i64);
            let (mut x, mut e) = (nums[i] as i64, range.min(k as i64));
            while e > 0 { if e & 1 > 0 { res = (res * x) % m }; e /= 2; x = (x * x) % m }
            k -= range as i32; if k < 1 { break }
        }; res as _
    }

```
```c++

    int maximumScore(vector<int>& a, int k) {
        int m = 1e9+7, n = size(a), res = 1; vector<int> sc(n), sl, sr, l(n, -1), r(n, n);
        for(int i = 0; i < n; ++i) {
            int x = a[i], c = 0; for (int p = 2; p * p <= x; p++)
                if (x % p == 0) { c++; while (x % p == 0) x /= p; }
            sc[i] = c + (x > 1);
        }
        for (int i = 0, j; i < size(a); i++) {
            for (j = size(a) - 1 - i; size(sl) && sc[sl.back()] < sc[i]; sl.pop_back());
            for (; size(sr) && sc[sr.back()] <= sc[j]; sr.pop_back());
            if (size(sl)) l[i] = sl.back(); sl.push_back(i);
            if (size(sr)) r[j] = sr.back(); sr.push_back(j);
        }
        vector<int> idx(size(a)); iota(begin(idx), end(idx), 0);
        sort(begin(idx), end(idx), [&](int i, int j) { return a[i] > a[j]; });
        for (int i : idx) {
            long range = 1LL * (i - l[i]) * (r[i] - i);
            long x = 1LL * a[i],  e = min(1L * k, range);
            for (; e; e /= 2, x = (x * x) % m) if (e & 1) res = (res * x) % m;
            if ((k -= range) < 1) break;
        }
        return res;
    }

```

