---
layout: leetcode-entry
title: "3464. Maximize the Distance Between Points on a Square"
permalink: "/leetcode/problem/2026-04-25-3464-maximize-the-distance-between-points-on-a-square/"
leetcode_ui: true
entry_slug: "2026-04-25-3464-maximize-the-distance-between-points-on-a-square"
---

[3464. Maximize the Distance Between Points on a Square](https://leetcode.com/problems/maximize-the-distance-between-points-on-a-square/solutions/8096752/kotlin-rust-by-samoylenkodmitry-po6k/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25042026-3464-maximize-the-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OkqspE2lZDA)

![25.04.2026.webp](/assets/leetcode_daily_images/25.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1339

#### Problem TLDR

Max of minimum distnace between selected k points on square perimeter

#### Intuition

```j
// if we use binary search the min distance:
    // how can we peek k elements?
    // how to peek first?
    // | * * * * * *     *   *  |(unrolled perimeter)
    //   |.........|             dist
    // so for every dist we should find the first item we take
    // and then check all others
    // dist - log(n)
    // every first - n
    // all others k fit? - n  (maybe this can be improved)
    // solution is n^2log(n)
    //
    // how to faster check k-fit?
    // we stop as soon as we take k elements from start
    // but what if dist is big and we have to skip n elements
    // how to jump to element by dist?
    //
    // 1 2 3   5 6   8 9
    // *                    (dist=5, how to jump to 5?)
    //         ^ binary search to this position?
    //        log(n)
    // so the solution now is nklog^2(n)
    // should be accepted
    //
    // now how to unroll?
    // (already 18 minutes just for thinking)
    // from 0,0 go clockwise
    // 0,0 - 0,1 - 0,2 - 1,2 - 2,2 - 2,1 - 2,0 - (1,0)?
    // or just take four edges then concat them
    // 26 minute
    // i have a doubt: the dist in binary search is not precise
    // 29 minute, lets go hints (they basically the same idea)
    //
    // except hint about selecting elements: no binary search?
    // 47 minute: wrong answer on test case side = 2 points = [[0,0],[1,2],[2,0],[2,2],[2,1]] k = 4
    // 2 instead of 1
    // 54 minute: i have error of duplicate on edges concatenations
    // 1:10 minute: gosh i spot that i can't use binary search on a cycled perimeter coordinates
    // ok let's give up, i can't spend more than 1 hour
```
Didn't solved myself.
Several aha-moments are required:
1. flatten the perimeter, every point has perimeter-distance to 0,0, sort by it
2. freeze the minimum at-most distance between points and binary search it
3. to optimally take k points we have to scan for starting point
4. to find next point from start we can use binary search
5. we have to check wrap-around between first taken and last taken point, 4s-(last-first)

#### Approach

* symmetry allows nice conversion to distance: top and left is (x+y), bottom and right is (perimeter - (x+y))
* inner binary search can be from previous position
* upper bound of distance is perimeter/k

#### Complexity

- Time complexity:
$$O(nklog^2(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxDistance(s: Int, p: Array<IntArray>, k: Int): Int {
        val P = 4L*s; var L = 0L; var H = P/k
        val e = p.map { (x,y) -> if (x==0||y==s)x+y+0L else P-x-y }.sorted()
        while (L <= H) {
            val M = (L + H) / 2
            if (p.indices.any { i -> var j = i; var c = 1
                do { j = e.binarySearch(e[j] + M,fromIndex=j); if (j < 0) j = -j-1 }
                while (j < e.size && e[j] - e[i] <= P - M && ++c < k)
                c == k
            }) L = M + 1 else H = M - 1
        }
        return H.toInt()
    }
```
```rust
    pub fn max_distance(s: i32, p: Vec<Vec<i32>>, k: i32) -> i32 {
        let (s, P, n, mut L, mut H) = (s as i64, 4*s as i64, p.len(), 0, 4*s as i64/k as i64);
        let mut e = p.iter().map(|v| { let(x,y)=(v[0]as i64, v[1]as i64);
                                      if x==0||y==s{x+y}else{P-x-y}}).sorted().collect_vec();
        while L <= H {
            let M = (L+H)/2;
            if (0..n).any(|i| (1..k).try_fold(i, |j,_| {
                let x = e[j..].partition_point(|&v| v < e[j]+M)+j;
                (x < n && e[x]-e[i] <= P-M).then_some(x)
            }).is_some()) { L = M + 1 } else { H = M - 1 }
        } H as i32
    }
```

