---
layout: leetcode-entry
title: "2353. Design a Food Rating System"
permalink: "/leetcode/problem/2025-09-17-2353-design-a-food-rating-system/"
leetcode_ui: true
entry_slug: "2025-09-17-2353-design-a-food-rating-system"
---

[2353. Design a Food Rating System](https://leetcode.com/problems/design-a-food-rating-system/description/) medium
[blog post](https://leetcode.com/problems/design-a-food-rating-system/solutions/7198751/kotlin-rust-by-samoylenkodmitry-giqg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17092025-2353-design-a-food-rating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nsZf1ZSXKB8)

![1.webp](/assets/leetcode_daily_images/5d9f7aa6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1115

#### Problem TLDR

Food rating system: change rating, peek highest by type #medium #ds

#### Intuition

Make `TreeSet` buckets for each cuisine.

#### Approach

* don't modify the rating while item in the `TreeSet`

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 340ms
class FoodRatings(val foods: Array<String>, val cuisines: Array<String>, val ratings: IntArray) {
    val foodToIdx = foods.indices.associate { foods[it] to it }
    val csToIdx = foods.indices.groupBy { cuisines[it] }.mapValues { (k, v) ->
                val q = TreeSet<Int>(compareBy({-ratings[it]}, {foods[it]})); q += v; q }
    fun changeRating(food: String, newRating: Int) {
        val i = foodToIdx[food]!!; val q = csToIdx[cuisines[i]]!!
        q -= i; ratings[i] = newRating; q += i
    }
    fun highestRated(cuisine: String) = foods[csToIdx[cuisine]!!.first()]
}

```
```rust

// 56ms
#[derive(Default)] struct FoodRatings(Vec<String>, Vec<String>, Vec<i32>, HashMap<String, usize>, HashMap<String, BTreeSet<(i32, String)>>);
impl FoodRatings {
    fn new(f: Vec<String>, c: Vec<String>, r: Vec<i32>) -> Self {
        let fi: HashMap<_,_> = (0..f.len()).map(|i| (f[i].clone(), i)).collect();
        let mut ct = HashMap::new();
        for i in 0..f.len() { ct.entry(c[i].clone()).or_insert(BTreeSet::new()).insert((-r[i],f[i].clone())); }
        Self(f, c, r, fi, ct)
    }
    fn change_rating(&mut self, f: String, v: i32) {
        let i = self.3[&f]; let c = &self.1[i];
        self.4.get_mut(c).unwrap().remove(&(-self.2[i],self.0[i].clone())); self.2[i] = v;
        self.4.get_mut(c).unwrap().insert((-v,self.0[i].clone()));
    }
    fn highest_rated(&self, c: String) -> String { self.4[&c].iter().next().unwrap().1.clone() }
}

```

