---
layout: leetcode-entry
title: "3186. Maximum Total Damage With Spell Casting"
permalink: "/leetcode/problem/2025-10-11-3186-maximum-total-damage-with-spell-casting/"
leetcode_ui: true
entry_slug: "2025-10-11-3186-maximum-total-damage-with-spell-casting"
---

[3186. Maximum Total Damage With Spell Casting](https://leetcode.com/problems/maximum-total-damage-with-spell-casting/description/) medium
[blog post](https://leetcode.com/problems/maximum-total-damage-with-spell-casting/solutions/7266675/kotlin-rust-by-samoylenkodmitry-0gqq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11102025-3186-maximum-total-damage?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6Gkh0w0Tvvc)

![baf56442-2656-48e6-a7b6-9e1d24b736a1 (1).webp](/assets/leetcode_daily_images/ce09808d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1139

#### Problem TLDR

Max sum of choosen values, skip v[i]+1+2-1-2 #medium #dp #sorting

#### Intuition

Thoughts process:

```j
    // 31 minute: greedy doesn't work
    // 6  5  4 3
    // 12 15 8 6
    // *  x  x *
    // x  *  x x
    // x  x  * x
    // *  x  x *

    // * x x x x * skip up to 4 elements
    // don't see greedy, try dp
    // 53 minute TLE
```
* sort and deduplicate
* dp[i] is the result for suffix [i..]
* dp[i] = max(take, notTake)
* if take, find the next index of p[i]+2

#### Approach

* to find next index you can use TreeMap.higherEntry(p+2)
* solution from u/votrubac/: dp[i+1] is the result *if* dp[i] is taken; dp[i]=max(dp[..j]), p[j]+2 is less than p[i]

#### Complexity

- Time complexity:
$$O(nlog(n))$$, solution uses TreeMap retrieval of `log(n)`. Sorting is nlog(n).

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 813ms
    fun maximumTotalDamage(p: IntArray): Long {
        val pc = p.groupBy { it }.mapValues { it.value.size }
        val pi = TreeMap<Int, Int>(); val dp = HashMap<Int,Long>()
        val keys = pc.keys.sorted(); for (i in keys.indices) pi[keys[i]] = i
        fun dfs(i: Int): Long = if (i==keys.size) 0L else  dp.getOrPut(i) {
            val p = keys[i]
            max(1L*p*pc[p]!! + (pi.higherEntry(p+2)?.let { dfs(it.value)}?:0), dfs(i+1))
        }
        return dfs(0)
    }

```
```rust

// 38ms
    pub fn maximum_total_damage(mut p:Vec<i32>)->i64{
        let (mut d, mut m) = (vec![(0,0)], 0);
        p.iter().sorted().chunk_by(|&x|x).into_iter().map(|(&v, g)| {
            while d.len() > 0 && d[0].1+2 < v as i64 { m = d.remove(0).0.max(m) }
            d.push((v as i64 * g.count() as i64 + m, v as i64)); d[d.len()-1].0
        }).max().unwrap()
    }

```

