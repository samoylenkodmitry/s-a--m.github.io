---
layout: leetcode-entry
title: "3160. Find the Number of Distinct Colors Among the Balls"
permalink: "/leetcode/problem/2025-02-07-3160-find-the-number-of-distinct-colors-among-the-balls/"
leetcode_ui: true
entry_slug: "2025-02-07-3160-find-the-number-of-distinct-colors-among-the-balls"
---

[3160. Find the Number of Distinct Colors Among the Balls](https://leetcode.com/problems/find-the-number-of-distinct-colors-among-the-balls/description/) medium
[blog post](https://leetcode.com/problems/find-the-number-of-distinct-colors-among-the-balls/solutions/6388528/kotlin-rust-by-samoylenkodmitry-7gav/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07022025-3160-find-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5rmQVVnIuhU)
![1.webp](/assets/leetcode_daily_images/2b353163.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/888

#### Problem TLDR

Running colors counter #medium #hashmap

#### Intuition

Store mappings: balls to colors, colors to balls. Results are colors size.

#### Approach

* we can only store frequencies of colors
* theoretically we can find a perfect hash function to just store [hash(x)] in min(limit, 10^5) array
* we can first collect uniq balls and colors, and use binary search in them instead of a hash map

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun queryResults(limit: Int, queries: Array<IntArray>) = {
        val f = HashMap<Int, Int>(); val bc = HashMap<Int, Int>()
        queries.map { (b, c) ->
            bc[b]?.let { f[it] = f[it]!! - 1; if (f[it]!! < 1) f -= it }
            bc[b] = c; f[c] = 1 + (f[c] ?: 0); f.size
        }
    }()

```
```rust

    pub fn query_results(limit: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let mut f = HashMap::new(); let mut bc = f.clone();
        queries.iter().map(|q| { let (b, c) = (q[0], q[1]);
            if let Some(&c) = bc.get(&b)
                { *f.entry(c).or_default() -= 1; if f[&c] < 1 { f.remove(&c); }}
            bc.insert(b, c); *f.entry(c).or_default() += 1; f.len() as i32
        }).collect()
    }

```
```c++

    vector<int> queryResults(int limit, vector<vector<int>>& q) {
        unordered_map<int, int>f, bc; vector<int> res;
        for (auto& p: q) bc.count(p[0]) && !--f[bc[p[0]]] && f.erase(bc[p[0]]),
            bc[p[0]] = p[1], f[p[1]]++, res.push_back(size(f));
        return res;
    }

```

