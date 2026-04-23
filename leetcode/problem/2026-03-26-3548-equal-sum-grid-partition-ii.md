---
layout: leetcode-entry
title: "3548. Equal Sum Grid Partition II"
permalink: "/leetcode/problem/2026-03-26-3548-equal-sum-grid-partition-ii/"
leetcode_ui: true
entry_slug: "2026-03-26-3548-equal-sum-grid-partition-ii"
---

[3548. Equal Sum Grid Partition II]() hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26032026-3548-equal-sum-grid-partition?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mGg9hSohWQI)

![26032026.webp](/assets/leetcode_daily_images/26.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1309

#### Problem TLDR

Cut matrix in half sums, can remove one cell #hard

#### Intuition

Didn't solve myself.

1. rotate 4 times to simplify the logic
2. lee intuition: store visited values in a hashset, calculate if suffix-prefix is in visited set
3. another intuition: store visited prefixes in a hashmap to cut positions, calculate if (total-value)/2 in visited prefixes

#### Approach

* reasoning about "not to disconnect" is the hardest part
* corners are allowed to cut
* single line grid is a corner case
* we are making horizontal cuts
* i==R && x%C==0 is the last row corners

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$ can be O(max(n,m))

#### Code

```kotlin
// 286ms
    fun canPartitionGrid(g: Array<IntArray>): Boolean {
        val l=g.map{it.map{1L*it}}; val t=l.sumOf{it.sum()}
        val x=l[0].indices.map{c->l.map{it[c]}}
        fun F(m:List<List<Long>>):Boolean {
            val R=m.size-1; val C=m[0].size-1; var p=0L; val M=HashMap<Long,Int>()
            return (0..R).any { y ->
                m[y].withIndex().any{ (x,c) ->
                    val i=M[(t-c)/2]; p += c
                    (t-c)%2==0L && i!=null && (if(C<1)y==i+1||y==R else i<R-1||x%C==0)
                } || y<R && { M[p]=y; p*2 == t }()
            }
        }
        return listOf(l,l.reversed(),x,x.reversed()).any(::F)
    }
```
```rust
// 128ms
    pub fn can_partition_grid(g: Vec<Vec<i32>>) -> bool {
        let l: Vec<Vec<_>> = g.iter().map(|r| r.iter().map(|&v| v as i64).collect()).collect();
        let t: i64 = l.iter().flatten().sum();
        let x: Vec<Vec<_>> = (0..l[0].len()).map(|c| l.iter().map(|r| r[c]).collect()).collect();
        let (mut lr, mut xr) = (l.clone(), x.clone()); lr.reverse(); xr.reverse();
        let f = |m: &Vec<Vec<i64>>| {
            let (r, c, mut p, mut map) = (m.len() - 1, m[0].len() - 1, 0, HashMap::new());
            (0..=r).any(|y| {
                m[y].iter().enumerate().any(|(x, &v)| {
                    let i = map.get(&((t - v) / 2)); p += v;
                    (t - v) % 2 == 0 && i.map_or(false, |&i| if c < 1 { y == i + 1 || y == r } else { i < r - 1 || x % c == 0 })
                }) || y < r && { map.insert(p, y); p * 2 == t }
            })
        };
        [&l, &lr, &x, &xr].into_iter().any(|m| f(m))
    }
```

