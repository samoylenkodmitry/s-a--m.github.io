---
layout: leetcode-entry
title: "2438. Range Product Queries of Powers"
permalink: "/leetcode/problem/2025-08-11-2438-range-product-queries-of-powers/"
leetcode_ui: true
entry_slug: "2025-08-11-2438-range-product-queries-of-powers"
---

[2438. Range Product Queries of Powers](https://leetcode.com/problems/range-product-queries-of-powers/description) medium
[blog post](https://leetcode.com/problems/range-product-queries-of-powers/solutions/7067025/kotlin-rust-by-samoylenkodmitry-gyxt/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11082025-2438-range-product-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/q8JvzYWRmpE)
![1.webp](/assets/leetcode_daily_images/c8a99d3b.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1077

#### Problem TLDR

Range product queries of powers of two from n #medium

#### Intuition

1. Known technique to split number to the powers of two: exponentiation, each set bit is the power of two
2. Trick observation: there are at most 30 of them, can brute-force each query
```j
   // 012345
    // 1248
    // 1111
    // ok how to do modulus subarray?
    // maybe use the fact they are all powers of two? (not many of them)

```

Interesting math optimization to O(nlog(30) + 30):
* product is `2^a * 2^b * 2^c = 2^(a+b+c)`
* so, we can use powers prefixes sum
* then product of range is `p_i_j = 2^(bits in i..j)`
* to calc `x^y %m` use math: `x^y = (x * x)^y/2 + x^y%2`

#### Approach

* use long, product can overflow even before `%m`
* optimization: precompute square matrix `30^2` for `result[from][to]`

#### Complexity

- Time complexity:
$$O(n)$$,

- Space complexity:
$$O(1)$$

#### Code

```kotlin [-Kotlin (71ms]

// 71ms
    fun productQueries(n: Int, q: Array<IntArray>): IntArray {
        val p = ArrayList<Int>(); val M = 1000000007L
        for (b in 0..30) if (n shr b and 1 != 0) p += 1 shl b
        return IntArray(q.size) {
            (q[it][0]..q[it][1]).fold(1L) { r, i -> (r * p[i]) % M }.toInt()
        }
    }

```
```kotlin [-19ms)]

// 19ms
    fun productQueries(n: Int, q: Array<IntArray>): IntArray {
        val p = IntArray(n.countOneBits() + 1); var i = 1; val M = 1000000007L
        for (b in 0..30) if (n shr b and 1 > 0) p[i] += b + p[i++ - 1]
        return IntArray(q.size) {
            val (l, r) = q[it]
            var k = p[r + 1] - p[l]
            var x = 1L; var b = 2L
            while (k > 0) {
                if (k and 1 > 0) x = (x * b) % M
                b = (b * b) % M; k = k shr 1
            }
            x.toInt()
        }
    }

```
```rust [-Rust 15ms]

// 15ms
    pub fn product_queries(n: i32, q: Vec<Vec<i32>>) -> Vec<i32> {
        let mut p = vec![]; let M = 1000000007i64;
        for b in 0..31 { if n >> b & 1 > 0 { p.push(1i64 << b) }}
        q.iter().map(|q| { let (s, e) = (q[0] as usize, q[1] as usize);
            (s..=e).fold(1, |r, i| (r * p[i]) % M) as i32
        }).collect()
    }

```
```c++ [-c++ 16ms]

// 16ms
    vector<int> productQueries(int n, vector<vector<int>>& q) {
        vector<int> p, r; int m = 1000000007;
        for (int b = 0; b < 31; ++b) if (n >> b & 1) p.push_back(1<<b);
        for (auto& q: q) {
            long x = 1L; for (int i = q[0]; i <= q[1]; ++i) x = (1L * x * p[i])%m;
            r.push_back((int) x);
        } return r;
    }

```
```python [-python 95ms]

// 95ms
    def productQueries(self, n: int, q: List[List[int]]) -> List[int]:
        m = 1_000_000_007
        p = [b for b in range(31) if (n >> b) & 1]
        pref = [0]
        for e in p:
            pref.append(pref[-1] + e)
        return [pow(2, pref[r+1] - pref[l], m) for l, r in q]

```

