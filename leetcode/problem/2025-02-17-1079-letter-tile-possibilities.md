---
layout: leetcode-entry
title: "1079. Letter Tile Possibilities"
permalink: "/leetcode/problem/2025-02-17-1079-letter-tile-possibilities/"
leetcode_ui: true
entry_slug: "2025-02-17-1079-letter-tile-possibilities"
---

[1079. Letter Tile Possibilities](https://leetcode.com/problems/letter-tile-possibilities/description/) medium
[blog post](https://leetcode.com/problems/letter-tile-possibilities/solutions/6432441/kotlin-rust-by-samoylenkodmitry-a1m7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17022025-1079-letter-tile-possibilities?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yHcaXpV6N_A)
![1.webp](/assets/leetcode_daily_images/d1d1347e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/898

#### Problem TLDR

Count uniq sequences from letters #medium #backtracking

#### Intuition

The problem size is 7 elements at most, the brute-force works: try to append every char, count ends at every position.

#### Approach

* modify input string or use the frequency counter
* duplicate letters is the corner case, use Set
* for the frequency solution, just try every char one-by-one if it exists
* memoization is possible: the result always depends of the input chars set

#### Complexity

- Time complexity:
$$O(n^n)$$ (7^7 = 823543, valid cases for ABCDEG = 13699, so the filtering matters)

- Space complexity:
$$O(n)$$ the recursion depth

#### Code

```kotlin

    fun numTilePossibilities(tiles: String): Int =
        tiles.toSet().sumBy { c ->
            1 + numTilePossibilities(tiles.replaceFirst("$c", ""))
        }

```
```rust

    pub fn num_tile_possibilities(tiles: String) -> i32 {
        let mut f = vec![0; 26];
        for b in tiles.bytes() { f[(b - b'A') as usize] += 1 }
        fn dfs(f: &mut Vec<i32>) -> i32 { (0..26).map(|b|
            if f[b] > 0 { f[b] -= 1; let r = 1 + dfs(f); f[b] += 1; r }
            else { 0 }).sum()
        }; dfs(&mut f)
    }

```
```c++

    int numTilePossibilities(string tiles) {
        int f[26] = {}; for (auto c: tiles) ++f[c - 'A'];
        auto d = [&](this const auto d) -> int {
            int cnt = 0; for (int i = 0; i < 26; ++i) if (f[i] > 0)
                --f[i], cnt += 1 + d(), ++f[i]; return cnt;
        };
        return d();
    }

```

