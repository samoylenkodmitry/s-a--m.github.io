---
layout: leetcode-entry
title: "2044. Count Number of Maximum Bitwise-OR Subsets"
permalink: "/leetcode/problem/2025-07-28-2044-count-number-of-maximum-bitwise-or-subsets/"
leetcode_ui: true
entry_slug: "2025-07-28-2044-count-number-of-maximum-bitwise-or-subsets"
---

[2044. Count Number of Maximum Bitwise-OR Subsets](https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/solutions/7014765/kotlin-rust-by-samoylenkodmitry-d6tz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28072025-2044-count-number-of-maximum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RRJ5xnh0FSc)
![1.webp](/assets/leetcode_daily_images/5a6230ce.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1063

#### Problem TLDR

Subsequencies with max OR #medium #bits #backtrack

#### Intuition

Problem size is small - 16, traverse all subsequencies.

The dp solution: dp[or] - number of subsets, dp[x | or_prev] += dp[or_prev]

#### Approach

* 0..2^16 are all possible bitmasks
* the brute-force is slower than backtracking, every time goes from start: n2^n vs 2^n
* calculate max on the go

#### Complexity

- Time complexity:
$$O(n2^n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 69ms
    fun countMaxOrSubsets(n: IntArray, o: Int = n.reduce(Int::or)) =
        (0..(1 shl n.size)).count { m ->
            o == n.indices.fold(0) { r, t -> r or (n[t] * (m shr t and 1)) }
        }

```
```kotlin

// 53ms
    fun countMaxOrSubsets(n: IntArray): Int {
        var res = 0; var max = 0
        for (m in 0..(1 shl n.size)) {
            val v = n.indices.fold(0) { r, t -> r or (n[t] * (m shr t and 1)) }
            if (v > max) { max = v; res = 1 } else if (v == max) ++res
        }
        return res
    }

```
```rust

// 14ms
    pub fn count_max_or_subsets(n: Vec<i32>) -> i32 {
        let (mut r, mut o) = (0, 0);
        for m in 0..=1 << n.len() as i32 {
            let v = (0..n.len()).fold(0, |r, t| r | n[t] * (m >> t & 1));
            if v > o { o = v; r = 0 }; if v == o { r += 1 }
        } r
    }

```
```c++

// 86ms
    int countMaxOrSubsets(vector<int>& n) {
        int max = 0, dp[1<<17]={1};
        for (int x: n) {
            for (int i = max; i >= 0; --i) dp[i | x] += dp[i];
            max |= x;
        } return dp[max];
    }

```
```python3

// 14ms
 def countMaxOrSubsets(self, n: List[int]) -> int:
        return (f:=cache(lambda i,v,o=reduce(or_,n):n[i:] and f(i+1,v)+f(i+1,v|n[i]) or v==o))(0,0)

```

